"""Dart extractor — tree-sitter AST walk with Flutter heuristic edges."""
from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.base import (
    _LANGUAGE_BUILTIN_GLOBALS,
    _file_stem,
    _make_id,
    _read_source_bytes,
    _read_text,
)

_DART_BUILTIN_TYPES = frozenset({
    "String", "int", "double", "bool", "num", "dynamic", "Object", "void",
    "List", "Map", "Set", "Future", "Stream", "Iterable", "Iterator",
    "Null", "Never", "Function", "Record", "Symbol", "Type",
})

# Annotations that carry no architecture signal. The compiler hints at the end
# (`@pragma`, `@optionalTypeArgs`, `@useResult`) sit on nearly every member of
# generated freezed / json_serializable code and would otherwise turn into hub
# nodes with thousands of `configures` edges.
_ANNOTATION_SKIP = frozenset({
    "override", "deprecated", "required", "protected", "mustCallSuper",
    "visibleForTesting", "immutable", "nonVirtual", "mustBeOverridden",
    "pragma", "optionalTypeArgs", "useResult",
})

_METHOD_NAME_SKIP = frozenset({
    "if", "for", "while", "switch", "catch", "return", "void", "dynamic",
    "final", "const", "get", "set", "var", "late", "typedef", "factory",
})

_MEMBER_SIGNATURES = ("function_signature", "getter_signature", "setter_signature")

_CTOR_SIGNATURES = (
    "constructor_signature",
    "constant_constructor_signature",
    "factory_constructor_signature",
    "redirecting_factory_constructor_signature",
)

_TYPE_DECLS = (
    "class_definition",
    "mixin_declaration",
    "enum_declaration",
    "extension_declaration",
    "extension_type_declaration",
    "type_alias",
)


def _line_of(node) -> int:
    return node.start_point[0] + 1


def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def _type_blacklist_ok(name: str) -> bool:
    return bool(name) and name not in _DART_BUILTIN_TYPES


def _blank_comments(root, source: bytes) -> bytes:
    """Return ``source`` with every comment blanked to spaces.

    Byte length and newlines are preserved, so node byte offsets and line
    numbers computed against the result match the original source. The regex
    heuristics run on this text so a commented-out ``emit(Loading())`` or
    ``context.push('/x')`` no longer mints an edge (the regex extractor scanned
    comment-stripped text too). String literals are kept: route paths live in
    them.
    """
    spans: list[tuple[int, int]] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in ("comment", "documentation_comment"):
            spans.append((n.start_byte, n.end_byte))
            continue
        stack.extend(n.children)
    if not spans:
        return source
    buf = bytearray(source)
    for start, end in spans:
        buf[start:end] = re.sub(rb"[^\n]", b" ", bytes(buf[start:end]))
    return bytes(buf)


def _type_groups(children, start: int = 0) -> list[tuple[list, object | None]]:
    """Read ``A, p.B<X>, C`` from a flat child list.

    Returns ``(type_identifier_nodes, type_arguments_node_or_None)`` for each
    top-level type (``p.B`` is two type_identifier nodes), so a type's own
    generic arguments are never reported as
    extra supertypes (``implements Cmp<Foo>`` is Cmp, not Cmp and Foo). Stops
    at the first child that cannot be part of the list (a body, another
    clause).
    """
    groups: list[tuple[list, object | None]] = []
    i = start
    n = len(children)
    while i < n:
        c = children[i]
        if c.type == "type_identifier":
            parts = [c]
            j = i + 1
            while j + 1 < n and children[j].type == "." and children[j + 1].type == "type_identifier":
                parts.append(children[j + 1])
                j += 2
            args = None
            if j < n and children[j].type == "type_arguments":
                args = children[j]
                j += 1
            groups.append((parts, args))
            i = j
            continue
        if c.type in (",", "<", ">", "?", "nullable_type"):
            i += 1
            continue
        if c.type in ("implements", "with", "extends", "on"):
            i += 1
            continue
        break
    return groups


def _decl_name_node(node):
    """The name node of a type declaration (None for an anonymous extension)."""
    if node.type == "type_alias":
        kids = node.children
        if any(c.type == "=" for c in kids):
            # typedef Name<T> = Type;
            return next((c for c in kids if c.type == "type_identifier"), None)
        # Legacy function typedef: `typedef int Compare(Object a, Object b);`
        # The name is the last type_identifier before the type parameters /
        # parameter list; any earlier one is the return type.
        last = None
        for c in kids:
            if c.type in ("formal_parameter_list", "type_parameters"):
                break
            if c.type == "type_identifier":
                last = c
        return last
    return next((c for c in node.children if c.type == "identifier"), None)


def _sig_name(sig, source: bytes) -> str | None:
    idents = [c for c in sig.children if c.type == "identifier"]
    return _read_text(idents[-1], source) if idents else None


def _list_names(id_list, source: bytes) -> list:
    """Identifier nodes declared by a field / variable declaration list."""
    out = []
    for ini in id_list.children:
        if ini.type == "identifier":
            out.append(ini)
        elif ini.type in ("initialized_identifier", "static_final_declaration"):
            ident = next((x for x in ini.children if x.type == "identifier"), None)
            if ident is not None:
                out.append(ident)
    return out


_ID_LISTS = ("initialized_identifier_list", "static_final_declaration_list", "identifier_list")
_FACTORY_SIGNATURES = ("factory_constructor_signature", "redirecting_factory_constructor_signature")


def _member_names(body, source: bytes) -> list[str]:
    """Every member name a type body declares (for the id-collision census)."""
    names: list[str | None] = []
    for child in body.children:
        if child.type not in ("method_signature", "declaration"):
            continue
        for c in child.children:
            if c.type in _MEMBER_SIGNATURES:
                names.append(_sig_name(c, source))
            elif c.type in _FACTORY_SIGNATURES:
                idents = [x for x in c.children if x.type == "identifier"]
                if len(idents) >= 2:
                    names.append(_read_text(idents[-1], source))
            elif c.type in _ID_LISTS:
                names.extend(_read_text(x, source) for x in _list_names(c, source))
    return [n for n in names if n]


def _top_level_names(root, source: bytes) -> list[str]:
    """Every name declared at file scope (types, functions, variables)."""
    names: list[str | None] = []
    for child in root.children:
        if child.type in _TYPE_DECLS:
            n = _decl_name_node(child)
            if n is not None:
                names.append(_read_text(n, source))
        elif child.type in _MEMBER_SIGNATURES:
            names.append(_sig_name(child, source))
        elif child.type in ("static_final_declaration_list", "initialized_identifier_list"):
            names.extend(_read_text(x, source) for x in _list_names(child, source))
    return [n for n in names if n]


def _collision_groups(names, nid_of) -> dict[str, set[str]]:
    """{plain id: names} for ids that two distinct names collapse into."""
    groups: dict[str, set[str]] = {}
    for name in names:
        groups.setdefault(nid_of(name), set()).add(name)
    return {nid: group for nid, group in groups.items() if len(group) >= 2}


def _qualified_type_before(node, source: bytes) -> str | None:
    """Rebuild auth.AuthService from sibling type_identifier / . nodes before node."""
    parent = node.parent
    if parent is None:
        return None
    siblings = list(parent.children)
    try:
        idx = siblings.index(node)
    except ValueError:
        return None
    i = idx - 1
    buf: list[str] = []
    while i >= 0:
        s = siblings[i]
        if s.type == "type_identifier":
            buf.append(_read_text(s, source))
            i -= 1
            continue
        if s.type in (".", "type_arguments"):
            i -= 1
            continue
        break
    if not buf:
        return None
    buf.reverse()
    return ".".join(buf)


def _apply_flutter_heuristics(body_text: str, body_start_line: int, owner_nid: str, add_node, add_edge) -> None:
    """Bloc / Riverpod / navigation edges from (comment-blanked) body source text."""
    if not body_text:
        return

    def _line_at(offset: int) -> int:
        return body_start_line + body_text[:offset].count("\n")

    for em in re.finditer(r"\bon<(\w+)>\s*\(", body_text):
        event_name = em.group(1)
        event_nid = _make_id(event_name)
        add_node(event_nid, event_name, source_file=None, line=None)
        add_edge(owner_nid, event_nid, "calls", line=_line_at(em.start(1)), context="bloc_event")

    for sm in re.finditer(r"\b(?:emit|yield)\s*\(?\s*(?:const\s+)?([A-Z]\w*)\b", body_text):
        state_name = sm.group(1)
        if state_name not in _DART_BUILTIN_TYPES:
            state_nid = _make_id(state_name)
            add_node(state_nid, state_name, source_file=None, line=None)
            add_edge(owner_nid, state_nid, "calls", line=_line_at(sm.start(1)), context="emit_state")

    for am in re.finditer(
        r"\b(?:\w*[Bb]loc\w*|context\.read<\w+>\(\))\.add\(\s*(?:const\s+)?([A-Z]\w*)\b",
        body_text,
    ):
        event_name = am.group(1)
        if event_name not in _DART_BUILTIN_TYPES:
            event_nid = _make_id(event_name)
            add_node(event_nid, event_name, source_file=None, line=None)
            add_edge(owner_nid, event_nid, "calls", line=_line_at(am.start(1)), context="bloc_add_event")

    for rm in re.finditer(r"\bref\.(?:watch|read|listen)\s*\(\s*(\w+)\b", body_text):
        provider_name = rm.group(1)
        provider_nid = _make_id(provider_name)
        add_node(provider_nid, provider_name, source_file=None, line=None)
        add_edge(
            owner_nid, provider_nid, "references",
            line=_line_at(rm.start(1)), context="riverpod_reference",
        )

    for bm in re.finditer(
        r"\bBloc(?:Builder|Listener|Consumer|Provider|Selector)\s*<\s*([a-zA-Z0-9_]+)\b",
        body_text,
    ):
        bloc_name = bm.group(1)
        if _type_blacklist_ok(bloc_name):
            bloc_nid = _make_id(bloc_name)
            add_node(bloc_nid, bloc_name, source_file=None, line=None)
            add_edge(
                owner_nid, bloc_nid, "references",
                line=_line_at(bm.start(1)), context="bloc_widget_binding",
            )

    for lm in re.finditer(r"\b(?:read|watch|select|of)\s*<([a-zA-Z0-9_]+)>", body_text):
        bloc_name = lm.group(1)
        if _type_blacklist_ok(bloc_name):
            bloc_nid = _make_id(bloc_name)
            add_node(bloc_nid, bloc_name, source_file=None, line=None)
            add_edge(
                owner_nid, bloc_nid, "references",
                line=_line_at(lm.start(1)), context="bloc_lookup",
            )

    for nm in re.finditer(
        r"\b(?:go|push|goNamed|pushNamed|replace|replaceNamed)\s*\(\s*(?:context\s*,\s*)?['\"]([a-zA-Z0-9_/?=&%-]+)['\"]",
        body_text,
    ):
        route_path = nm.group(1)
        route_nid = _make_id(
            "route",
            route_path.replace("/", "_").replace("?", "_").replace("=", "_").replace("&", "_"),
        )
        add_node(route_nid, f"Route {route_path}", ftype="concept", source_file=None, line=None)
        add_edge(owner_nid, route_nid, "navigates", line=_line_at(nm.start(1)), context="route_path")

    for cm in re.finditer(
        r"\b(?:go|push|goNamed|pushNamed|replace|replaceNamed)\s*\(\s*(?:context\s*,\s*)?([A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+)",
        body_text,
    ):
        route_const = cm.group(1)
        route_nid = _make_id("route", route_const.replace(".", "_"))
        add_node(route_nid, route_const, ftype="concept", source_file=None, line=None)
        add_edge(owner_nid, route_nid, "navigates", line=_line_at(cm.start(1)), context="route_const")

    for om in re.finditer(
        r"\b(?:push|replace)\s*\(\s*(?:context\s*,\s*)?.*?\b([A-Z]\w*(?:Route|Screen|Page))\b",
        body_text,
    ):
        route_class = om.group(1)
        route_nid = _make_id(route_class)
        add_node(route_nid, route_class, source_file=None, line=None)
        add_edge(owner_nid, route_nid, "navigates", line=_line_at(om.start(1)), context="route_object")


def extract_dart(path: Path) -> dict:
    """Extract Dart/Flutter symbols via tree-sitter AST (+ Flutter body heuristics)."""
    try:
        import tree_sitter_dart as tsdart
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-dart not installed"}
    # Lazy: graphify.extractors imports this module while it initialises.
    from graphify.extractors.engine import _underscore_salted_nid

    try:
        source = _read_source_bytes(path)
    except OSError:
        return {"nodes": [], "edges": [], "error": f"cannot read {path}"}

    try:
        parser = Parser(Language(tsdart.language()))
        root = parser.parse(source).root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    # Same bytes with comments blanked: for the regex heuristics only. The AST
    # walk and node text keep reading `source`.
    clean = _blank_comments(root, source)

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    node_by_id: dict[str, dict] = {}
    seen_ids = node_by_id.keys()
    # caller nid, body node. Calls inside closures and local functions are
    # credited to the enclosing declaration, which is the graph node they run in.
    function_bodies: list[tuple[str, object]] = []
    # Call-resolution indexes (see the call pass at the end).
    top_callables: dict[str, list[str]] = {}       # label -> top-level functions/types
    class_members: dict[str, dict[str, str]] = {}  # type nid -> {member label: nid}
    member_owner: dict[str, str] = {}              # member nid -> type nid
    member_by_label: dict[str, list[str]] = {}     # label -> every member nid

    def add_node(
        nid: str,
        label: str,
        *,
        ftype: str = "code",
        source_file: str | None = str_path,
        line: int | None = None,
    ) -> bool:
        if nid in node_by_id:
            return False
        node = {
            "id": nid,
            "label": label,
            "file_type": ftype,
            "source_file": source_file,
            "source_location": f"L{line}" if line is not None else None,
        }
        node_by_id[nid] = node
        nodes.append(node)
        return True

    def add_edge(
        src: str,
        tgt: str,
        relation: str,
        *,
        line: int | None = None,
        weight: float = 1.0,
        context: str | None = None,
    ) -> None:
        edge = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "source_file": str_path,
            "source_location": f"L{line}" if line is not None else None,
            "weight": weight,
        }
        if context:
            edge["context"] = context
        edges.append(edge)

    def ensure_external(name: str) -> str:
        nid = _make_id(name)
        add_node(nid, name, source_file=None)
        return nid

    def register_member(type_nid: str, label: str, nid: str) -> None:
        class_members.setdefault(type_nid, {}).setdefault(label, nid)
        member_owner[nid] = type_nid
        member_by_label.setdefault(label, []).append(nid)

    # --- part of redirect ---
    # Every scanned file keeps its own file node. For a part file the symbols
    # still belong to the library (file_nid); the part's node is linked from the
    # library and to its symbols at the end of extraction.
    own_nid = _make_id(str_path)
    file_nid = own_nid
    is_part = False
    part_of_line: int | None = None
    for top in root.children:
        if top.type != "part_of_directive":
            continue
        uri = next((c for c in top.children if c.type == "uri"), None)
        part_ref = _strip_quotes(_read_text(uri, source)) if uri is not None else ""
        if part_ref.endswith(".dart"):
            try:
                parent_path = (path.parent / part_ref).resolve()
                if parent_path.exists():
                    stem = _file_stem(parent_path)
                    file_nid = _make_id(str(parent_path))
                    is_part = True
                    part_of_line = _line_of(uri)
            except Exception:
                pass
        break

    add_node(own_nid, path.name, source_file=str_path, line=None)

    # `_make_id` drops `_` and `$` and lowercases, so freezed's `mixin _$AuthVm`
    # and `class _AuthVm`, or a field `_items` and its getter `items`, collapse
    # into one id and the second declaration vanished. Same rule as the
    # Python / C# / JS extractors (#3302, #4415, #4475): a unique public name
    # keeps the plain id, every other colliding name gets a salted one, so the
    # result never depends on declaration order. The census is per scope: file
    # scope here, each type body in handle_class_body.
    _dart_private = ("_", "$")
    top_groups = _collision_groups(_top_level_names(root, source), lambda n: _make_id(stem, n))

    def top_nid(name: str) -> str:
        return _underscore_salted_nid(_make_id(stem, name), name, top_groups, _dart_private)

    def member_nid(type_nid: str, name: str, groups: dict) -> str:
        return _underscore_salted_nid(_make_id(type_nid, name), name, groups, _dart_private)

    def emit_annotation_edges(ann_node, target_nid: str, target_name: str, target_kind: str) -> None:
        name = None
        for c in ann_node.children:
            if c.type == "identifier":
                name = _read_text(c, source)
                break
        if not name or name in _ANNOTATION_SKIP:
            return
        line = _line_of(ann_node)
        annotation_nid = _make_id("annotation", name.lower())
        add_node(annotation_nid, f"@{name}", ftype="concept", source_file=None, line=None)
        add_edge(target_nid, annotation_nid, "configures", line=line)
        if name.lower() == "riverpod":
            if target_kind == "class":
                provider_name = (
                    target_name[0].lower() + target_name[1:] + "Provider"
                    if len(target_name) > 1
                    else target_name.lower() + "Provider"
                )
            else:
                provider_name = target_name + "Provider"
            provider_nid = _make_id(provider_name)
            add_node(provider_nid, provider_name, ftype="concept", source_file=str_path, line=line)
            add_edge(target_nid, provider_nid, "defines", line=line, context="riverpod_provider")

    def group_name(parts) -> str:
        return ".".join(_read_text(p, source) for p in parts)

    def emit_supertype(type_nid: str, parts, args, relation: str, line: int, *, with_arg_refs: bool) -> None:
        name = group_name(parts)
        last = name.split(".")[-1]
        if relation != "inherits" and not (_type_blacklist_ok(last) or name == "Object"):
            return
        add_edge(type_nid, ensure_external(name), relation, line=line)
        if with_arg_refs and args is not None:
            # MyBloc extends Bloc<MyEvent, MyState> references MyEvent, MyState.
            for arg_parts, _ in _type_groups(args.children):
                arg = group_name(arg_parts)
                if _type_blacklist_ok(arg.split(".")[-1]):
                    add_edge(type_nid, ensure_external(arg), "references", line=line)

    def emit_heritage(type_nid: str, type_node, line: int) -> None:
        """extends / with / implements / mixin `on` clauses of a type declaration."""
        for child in type_node.children:
            if child.type == "superclass":
                kids = child.children
                groups = _type_groups(kids)
                if groups:
                    parts, args = groups[0]
                    emit_supertype(type_nid, parts, args, "inherits", line, with_arg_refs=True)
                for c in kids:
                    if c.type == "mixins":
                        for parts, args in _type_groups(c.children):
                            emit_supertype(type_nid, parts, args, "mixes_in", line, with_arg_refs=False)
            elif child.type == "mixins":
                for parts, args in _type_groups(child.children):
                    emit_supertype(type_nid, parts, args, "mixes_in", line, with_arg_refs=False)
            elif child.type == "interfaces":
                for parts, args in _type_groups(child.children):
                    emit_supertype(type_nid, parts, args, "implements", line, with_arg_refs=False)

        # A mixin's `on` clause is a bare sibling list and may name several
        # superclass constraints (`mixin M on A, B<C> {}`); each one inherits.
        if type_node.type == "mixin_declaration":
            kids = type_node.children
            for i, c in enumerate(kids):
                if c.type == "on":
                    for parts, args in _type_groups(kids, i + 1):
                        emit_supertype(type_nid, parts, args, "inherits", line, with_arg_refs=True)
                    break

    def add_member(type_nid: str, groups: dict, fname: str, line: int, pending_annotations: list) -> str | None:
        if not fname or fname in _METHOD_NAME_SKIP or re.match(r"^[A-Z]", fname):
            return None
        mnid = member_nid(type_nid, fname, groups)
        if add_node(mnid, fname, line=line):
            add_edge(type_nid, mnid, "defines", line=line)
            # Also file -> defines, so label-based tools still see membership.
            add_edge(file_nid, mnid, "defines", line=line)
            register_member(type_nid, fname, mnid)
        for ann in pending_annotations:
            emit_annotation_edges(ann, mnid, fname, "function")
        return mnid

    def add_fields(type_nid: str, groups: dict, decl, id_list) -> None:
        parts: list[str] = []
        for c in decl.children:
            if c is id_list:
                break
            if c.type == "type_identifier":
                parts.append(_read_text(c, source))
        var_type = ".".join(parts) if parts else None
        for ident in _list_names(id_list, source):
            vname = _read_text(ident, source)
            if not vname or vname in _METHOD_NAME_SKIP:
                continue
            line = _line_of(ident)
            vnid = member_nid(type_nid, vname, groups)
            if add_node(vnid, vname, line=line):
                add_edge(type_nid, vnid, "defines", line=line)
                add_edge(file_nid, vnid, "defines", line=line)
            if var_type:
                clean_type = var_type.split(".")[-1]
                if _type_blacklist_ok(clean_type):
                    add_edge(file_nid, ensure_external(clean_type), "references", line=line, context="variable_type")

    def handle_class_body(body_node, type_nid: str) -> None:
        if body_node is None:
            return
        groups = _collision_groups(_member_names(body_node, source), lambda n: _make_id(type_nid, n))
        children = body_node.children
        i = 0
        pending_annotations: list = []
        while i < len(children):
            child = children[i]
            if child.type == "annotation":
                pending_annotations.append(child)
                i += 1
                continue

            if child.type == "declaration":
                # Field, bodiless constructor, or abstract/external member.
                sig = next((c for c in child.children if c.type in _CTOR_SIGNATURES or c.type in _MEMBER_SIGNATURES), None)
                if sig is not None and sig.type in ("factory_constructor_signature", "redirecting_factory_constructor_signature"):
                    idents = [c for c in sig.children if c.type == "identifier"]
                    # factory MyService.fromJson -> the last identifier is fromJson
                    if len(idents) >= 2:
                        add_member(type_nid, groups, _read_text(idents[-1], source), _line_of(sig), pending_annotations)
                elif sig is not None and sig.type in _MEMBER_SIGNATURES:
                    # `Future<bool> openLogin();` in an interface / abstract class.
                    idents = [c for c in sig.children if c.type == "identifier"]
                    if idents:
                        add_member(type_nid, groups, _read_text(idents[-1], source), _line_of(sig), pending_annotations)
                elif sig is None:
                    id_list = next(
                        (c for c in child.children
                         if c.type in ("initialized_identifier_list", "static_final_declaration_list", "identifier_list")),
                        None,
                    )
                    if id_list is not None:
                        add_fields(type_nid, groups, child, id_list)
                pending_annotations = []
                i += 1
                continue

            if child.type == "method_signature":
                body = children[i + 1] if i + 1 < len(children) and children[i + 1].type == "function_body" else None
                fname = None
                is_ctor = False
                for c in child.children:
                    if c.type in ("factory_constructor_signature", "redirecting_factory_constructor_signature"):
                        # `factory A.fromJson()` is a member node; an unnamed
                        # `factory A()` is credited to the type like a constructor.
                        is_ctor = True
                        idents = [x for x in c.children if x.type == "identifier"]
                        if len(idents) >= 2:
                            fname = _read_text(idents[-1], source)
                    elif c.type in ("constructor_signature", "constant_constructor_signature"):
                        is_ctor = True
                    elif c.type in _MEMBER_SIGNATURES:
                        idents = [x for x in c.children if x.type == "identifier"]
                        if idents:
                            fname = _read_text(idents[-1], source)
                owner = add_member(type_nid, groups, fname, _line_of(child), pending_annotations) if fname else None
                if owner is None and is_ctor:
                    # A generative constructor is not its own node; its body
                    # (Bloc `on<Event>` registrations, init calls) runs in the type.
                    owner = type_nid
                if body is not None and owner is not None:
                    function_bodies.append((owner, body))
                    if owner != type_nid:
                        _apply_flutter_heuristics(_read_text(body, clean), _line_of(body), owner, add_node, add_edge)
                pending_annotations = []
                i += 2 if body is not None else 1
                continue

            pending_annotations = []
            i += 1

        # Whole-body heuristics on the type (Bloc ctor registrations live in constructors).
        _apply_flutter_heuristics(_read_text(body_node, clean), _line_of(body_node), type_nid, add_node, add_edge)

    def extension_target(node):
        kids = node.children
        for i, c in enumerate(kids):
            if c.type == "on":
                groups = _type_groups(kids, i + 1)
                if groups:
                    parts, _ = groups[0]
                    return group_name(parts), parts[0]
                return None
        return None

    def handle_type_decl(node) -> tuple[str, str] | None:
        ext = extension_target(node) if node.type == "extension_declaration" else None
        name_node = _decl_name_node(node)
        if name_node is not None:
            name = label = _read_text(name_node, source)
            line = _line_of(name_node)
            type_nid = top_nid(name)
        elif ext is not None:
            # Anonymous `extension on T { ... }`.
            name = f"{stem}_anonymous_extension"
            label = f"Extension on {ext[0]}"
            line = _line_of(ext[1])
            type_nid = _make_id(stem, name)
            if type_nid in seen_ids:
                name = f"{stem}_anonymous_extension_{ext[0]}"
                type_nid = _make_id(stem, name)
        else:
            return None

        if add_node(type_nid, label, line=line):
            add_edge(file_nid, type_nid, "defines", line=line)
        if node.type != "type_alias":
            top_callables.setdefault(label, []).append(type_nid)

        for c in node.children:
            if c.type == "annotation":
                emit_annotation_edges(c, type_nid, name, "class")

        emit_heritage(type_nid, node, line)

        if node.type == "type_alias" and any(c.type == "=" for c in node.children):
            # typedef JsonMap = p.Map<...>; references the aliased named type.
            # A function-type alias (`= R Function(int)`) nests its return type
            # under function_type, so it references nothing here.
            kids = node.children
            eq = next(i for i, c in enumerate(kids) if c.type == "=")
            groups = _type_groups(kids, eq + 1)
            if groups:
                target = group_name(groups[0][0]).split(".")[-1]
                if _type_blacklist_ok(target):
                    add_edge(type_nid, ensure_external(target), "references", line=line, context="typedef")

        body = next((c for c in node.children if c.type in ("class_body", "enum_body", "extension_body")), None)
        if body is not None:
            handle_class_body(body, type_nid)

        if ext is not None:
            add_edge(type_nid, ensure_external(ext[0]), "extends", line=line)
        return type_nid, name

    def handle_top_function(sig_node, body_node, pending_annotations: list) -> None:
        idents = [c for c in sig_node.children if c.type == "identifier"]
        if not idents:
            return
        fname = _read_text(idents[-1], source)
        if not fname or fname in _METHOD_NAME_SKIP or re.match(r"^[A-Z]", fname):
            return
        line = _line_of(idents[-1])
        fnid = top_nid(fname)
        if add_node(fnid, fname, line=line):
            add_edge(file_nid, fnid, "defines", line=line)
            top_callables.setdefault(fname, []).append(fnid)
        for ann in pending_annotations:
            emit_annotation_edges(ann, fnid, fname, "function")
        if body_node is not None:
            function_bodies.append((fnid, body_node))
            _apply_flutter_heuristics(_read_text(body_node, clean), _line_of(body_node), fnid, add_node, add_edge)

    def handle_import_export(node) -> None:
        relation = "exports" if any(c.type == "library_export" for c in node.children) else "imports"
        uris: list[tuple[object, str | None]] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "configurable_uri":
                # `import 'x_stub.dart' if (dart.library.io) 'x_io.dart';` — the
                # default URI keeps its plain edge; every platform branch is
                # emitted too, tagged conditional_uri. A string inside the
                # condition (`== 'prod'`) is not a URI.
                for c in n.children:
                    if c.type == "uri":
                        uris.append((c, None))
                    elif c.type == "configuration_uri":
                        u = next((x for x in c.children if x.type == "uri"), None)
                        if u is not None:
                            uris.append((u, "conditional_uri"))
                break
            if n.type == "uri":
                uris.append((n, None))
                break
            stack.extend(reversed(n.children))
        for uri_node, context in uris:
            uri = _strip_quotes(_read_text(uri_node, source))
            if not uri:
                continue
            tgt = ensure_external(uri)
            node_by_id[tgt]["label"] = uri
            add_edge(file_nid, tgt, relation, line=_line_of(uri_node), context=context)

    def handle_top_level_vars(program) -> None:
        """Scan program children for top-level variable declarations."""
        children = list(program.children)
        i = 0
        while i < len(children):
            c = children[i]

            # static_final_declaration_list after final/const/late/type
            if c.type == "static_final_declaration_list":
                for decl in c.children:
                    if decl.type != "static_final_declaration":
                        continue
                    ident = next((x for x in decl.children if x.type == "identifier"), None)
                    if not ident:
                        continue
                    vname = _read_text(ident, source)
                    if not vname or vname in _METHOD_NAME_SKIP:
                        continue
                    line = _line_of(ident)
                    vnid = top_nid(vname)
                    if add_node(vnid, vname, line=line):
                        add_edge(file_nid, vnid, "defines", line=line)
                i += 1
                continue

            if c.type == "initialized_identifier_list":
                # late final Type name; or pattern leftovers
                for ini in c.children:
                    if ini.type != "initialized_identifier":
                        continue
                    ident = next((x for x in ini.children if x.type == "identifier"), None)
                    if not ident:
                        continue
                    vname = _read_text(ident, source)
                    if not vname or vname in _METHOD_NAME_SKIP:
                        continue
                    # Skip PascalCase type names mistaken as vars
                    if re.match(r"^[A-Z]", vname):
                        continue
                    line = _line_of(ident)
                    vnid = top_nid(vname)
                    if add_node(vnid, vname, line=line):
                        add_edge(file_nid, vnid, "defines", line=line)
                    var_type = _qualified_type_before(c, source)
                    if var_type:
                        clean_type = var_type.split(".")[-1]
                        if _type_blacklist_ok(clean_type):
                            add_edge(
                                file_nid, ensure_external(clean_type), "references",
                                line=line, context="variable_type",
                            )
                i += 1
                continue

            # Pattern: ERROR(var) + record_type (recA, recB)
            if c.type in ("ERROR", "inferred_type"):
                j = i + 1
                while j < len(children) and children[j].type in ("ERROR", "inferred_type"):
                    j += 1
                if j < len(children) and children[j].type == "record_type":
                    for field in children[j].children:
                        if field.type != "record_type_field":
                            continue
                        idents = [x for x in field.children if x.type == "identifier"]
                        if not idents:
                            idents = [x for x in field.children if x.type == "type_identifier"]
                        for ident in idents:
                            vname = _read_text(ident, source)
                            if vname and re.match(r"^[a-z_]\w*$", vname):
                                line = _line_of(ident)
                                vnid = top_nid(vname)
                                if add_node(vnid, vname, line=line):
                                    add_edge(file_nid, vnid, "defines", line=line)
                i += 1
                continue

            i += 1

        # Regex supplement for object-pattern destructuring the grammar mishandles:
        # var User(name: myVar, age: myAge) = user;
        src_text = clean.decode("utf-8", errors="replace")
        for m in re.finditer(r"^\s*(?:var|final|const)\s+[A-Z]\w*\(([^)]+)\)\s*=", src_text, re.MULTILINE):
            for part in m.group(1).split(","):
                part = part.strip()
                name = part.split(":")[-1].strip() if ":" in part else part
                if re.match(r"^[a-z_]\w*$", name):
                    line = src_text[: m.start()].count("\n") + 1
                    vnid = top_nid(name)
                    if add_node(vnid, name, line=line):
                        add_edge(file_nid, vnid, "defines", line=line)

    # --- walk program ---
    pending_annotations: list = []
    children = list(root.children)
    i = 0
    while i < len(children):
        child = children[i]

        if child.type == "annotation":
            # Belongs to the next class/function.
            pending_annotations.append(child)
            i += 1
            continue

        if child.type in _TYPE_DECLS:
            decl = handle_type_decl(child)
            if decl is not None:
                type_nid, name = decl
                for ann in pending_annotations:
                    if ann.parent is not child:  # children were emitted already
                        emit_annotation_edges(ann, type_nid, name, "class")
            pending_annotations = []
            i += 1
            continue

        if child.type in _MEMBER_SIGNATURES:
            # Top-level function, getter or setter.
            body = children[i + 1] if i + 1 < len(children) and children[i + 1].type == "function_body" else None
            handle_top_function(child, body, pending_annotations)
            pending_annotations = []
            i += 2 if body is not None else 1
            continue

        if child.type == "import_or_export":
            handle_import_export(child)
            pending_annotations = []
            i += 1
            continue

        # Don't clear annotations on trivia
        if child.type in (";", "comment", "documentation_comment"):
            i += 1
            continue

        if child.is_named and child.type not in (
            "final_builtin", "const_builtin", "late", "type_identifier",
            "type_arguments", "static_final_declaration_list",
            "initialized_identifier_list", "record_type", "inferred_type",
            "ERROR", "part_of_directive", "part_directive", "library_name",
        ):
            pending_annotations = []

        i += 1

    handle_top_level_vars(root)

    # File-level generic type lookups (`get<AuthService>()`), once per call site,
    # over comment-blanked text.
    src_text = clean.decode("utf-8", errors="replace")
    generic_call_pattern = r"\b\w+<([a-zA-Z0-9_.]+(?:<[a-zA-Z0-9_.,\s<>]+>)?)\s*>\s*\("
    for m in re.finditer(generic_call_pattern, src_text):
        type_name = m.group(1).split(".")[-1].strip()
        clean_name = type_name.split("<")[0].strip()
        if _type_blacklist_ok(clean_name):
            target_nid = ensure_external(clean_name)
            node_by_id[target_nid]["label"] = clean_name
            line = src_text[: m.start(1)].count("\n") + 1
            add_edge(file_nid, target_nid, "references", line=line, context="type_lookup")

    # --- call-graph second pass ---
    # A call binds only when its target is unambiguous in this file:
    #   * a bare `foo()` / `this.foo()` inside a member prefers the caller's
    #     own type's member `foo` (implicit `this`);
    #   * otherwise a bare call binds to a unique top-level function or type
    #     (`Greeter(...)` is a constructor call);
    #   * a member call `x.foo()` binds to the one member named `foo` in this
    #     file. With two candidates (a StatefulWidget and its State both
    #     declaring `build`) nothing is emitted: a wrong EXTRACTED edge is
    #     worse than a missing one.
    # Unknown callees become raw_calls. They stay flagged as member calls so
    # the corpus-wide bare-name resolver leaves them alone, as before.
    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def resolve(callee: str, caller_nid: str, is_member: bool, on_this: bool) -> tuple[str | None, bool]:
        """Return (target nid or None, known) where known=False means no local candidate."""
        owner = member_owner.get(caller_nid) or (caller_nid if caller_nid in class_members else None)
        if (not is_member or on_this) and owner is not None:
            own = class_members.get(owner, {}).get(callee)
            if own is not None:
                return own, True
        if not is_member:
            cands = top_callables.get(callee, [])
            if len(cands) == 1:
                return cands[0], True
            if cands:
                return None, True
            if owner is None:
                # Top-level code cannot call a member without a receiver.
                return None, bool(member_by_label.get(callee))
        # A member call, or a bare call inside a member that its own type does
        # not declare (inherited from a same-file supertype).
        cands = member_by_label.get(callee, [])
        if len(cands) == 1:
            return cands[0], True
        return None, bool(cands)

    def emit_call(callee: str, call_node, caller_nid: str, is_member: bool, on_this: bool) -> None:
        if not callee or callee in _LANGUAGE_BUILTIN_GLOBALS or callee in _METHOD_NAME_SKIP:
            return
        tgt, known = resolve(callee, caller_nid, is_member, on_this)
        line = _line_of(call_node)
        if tgt is not None:
            if tgt == caller_nid:
                return
            pair = (caller_nid, tgt)
            if pair in seen_call_pairs:
                return
            seen_call_pairs.add(pair)
            edges.append({
                "source": caller_nid,
                "target": tgt,
                "relation": "calls",
                "context": "call",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": str_path,
                "source_location": f"L{line}",
                "weight": 1.0,
            })
        elif not known:
            raw_calls.append({
                "caller_nid": caller_nid,
                "callee": callee,
                "is_member_call": True,
                "language": "dart",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def walk_calls(body_node, caller_nid: str) -> None:
        # The grammar flattens a call chain into siblings: `a.b.c(1)` is
        # identifier(a) selector(.b) selector(.c) selector(argument_part). A
        # selector whose first child is argument_part is a call of whatever
        # precedes it: an identifier (bare call) or a `.name` / `?.name`
        # selector (member call). This finds calls in every position:
        # statements, initializers, conditions, arguments, arrow bodies,
        # closures and local functions.
        stack = [body_node]
        while stack:
            n = stack.pop()
            kids = n.children
            for k, c in enumerate(kids):
                if k and c.type == "selector" and c.named_child_count and c.named_children[0].type == "argument_part":
                    prev = kids[k - 1]
                    if prev.type == "identifier":
                        emit_call(_read_text(prev, source), prev, caller_nid, False, False)
                    elif prev.type == "selector" and prev.named_child_count:
                        sel = prev.named_children[0]
                        if sel.type in ("unconditional_assignable_selector", "conditional_assignable_selector"):
                            ident = next((x for x in sel.children if x.type == "identifier"), None)
                            if ident is not None:
                                on_this = k == 2 and kids[0].type == "this"
                                emit_call(_read_text(ident, source), ident, caller_nid, True, on_this)
                stack.append(c)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    if is_part:
        # The library includes this part; the part physically contains the
        # declarations the library defines from it. Ids stay in the library's
        # namespace, so the library is not split in two.
        part_edges = [{**e, "source": own_nid, "relation": "contains"}
                      for e in edges
                      if e["source"] == file_nid and e["relation"] == "defines"]
        add_edge(file_nid, own_nid, "includes", line=part_of_line)
        edges.extend(part_edges)

    return {"nodes": nodes, "edges": edges, "raw_calls": raw_calls}
