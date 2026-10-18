"""Typed scene-binding contracts and MuJoCo validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Literal, Mapping, Sequence

import mujoco
import numpy as np

from .json_values import to_json_value


ProgramKind = Literal["primitive", "strategy"]
RoleValue = str | tuple[str, ...]
InvariantCheck = Callable[
    [mujoco.MjModel, "SceneBinding"],
    Sequence["BindingIssue"] | Sequence[str],
]


class EntityKind(str, Enum):
    """MuJoCo named-object categories supported by scene bindings."""

    BODY = "body"
    GEOM = "geom"
    SITE = "site"
    JOINT = "joint"
    ACTUATOR = "actuator"


class RelationKind(str, Enum):
    """Declarative relationships that can be checked without program code."""

    DISTINCT_ENTITIES = "distinct_entities"
    SAME_BODY = "same_body"
    DIFFERENT_BODIES = "different_bodies"
    DESCENDANT_BODY_OF = "descendant_body_of"


class ParameterType(str, Enum):
    """JSON value categories allowed for primitive effect-goal parameters."""

    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"
    VECTOR = "vector"


@dataclass(frozen=True)
class ParameterRequirement:
    """One intrinsic effect-goal input accepted by a learned primitive.

    Parameters are deliberately narrower than arbitrary JSON. They describe
    effect variables such as requested displacement, direction, or angle whose
    values may come from a direct primitive invocation, a demonstrated construction
    example, or a strategy connecting constraint. Measured geometry and motion
    policy coefficients remain in learned code.
    """

    name: str
    value_type: ParameterType
    description: str
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    vector_length: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name cannot be empty")
        if not isinstance(self.value_type, ParameterType):
            raise TypeError("value_type must be a ParameterType")
        if not self.description:
            raise ValueError("parameter description cannot be empty")
        object.__setattr__(self, "choices", tuple(self.choices))
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("parameter minimum cannot exceed maximum")
        if self.choices and self.value_type is not ParameterType.STRING:
            raise ValueError("choices only apply to string parameters")
        if len(self.choices) != len(set(self.choices)) or any(
            not isinstance(item, str) or not item for item in self.choices
        ):
            raise ValueError("parameter choices must be unique non-empty strings")
        if (
            self.minimum is not None or self.maximum is not None
        ) and self.value_type not in {
            ParameterType.FLOAT,
            ParameterType.INTEGER,
            ParameterType.VECTOR,
        }:
            raise ValueError("minimum and maximum apply only to numeric parameters")
        if self.vector_length is not None:
            if self.value_type is not ParameterType.VECTOR:
                raise ValueError("vector_length only applies to vector parameters")
            if self.vector_length <= 0:
                raise ValueError("vector_length must be positive")
        if self.value_type is ParameterType.VECTOR and self.vector_length is None:
            raise ValueError("vector parameters require vector_length")


@dataclass(frozen=True)
class RelationalInvariant:
    """One relationship between two semantic roles."""

    relation: RelationKind
    left_role: str
    right_role: str
    description: str

    def __post_init__(self) -> None:
        if not self.left_role or not self.right_role:
            raise ValueError("relational invariant roles cannot be empty")
        if self.left_role == self.right_role:
            raise ValueError("relational invariant roles must be different")
        if not self.description:
            raise ValueError("relational invariant description cannot be empty")


@dataclass(frozen=True)
class RoleRequirement:
    """One semantic role that a program requires from a concrete scene."""

    role: str
    kind: EntityKind
    description: str
    min_count: int = 1
    max_count: int | None = 1
    allowed_types: tuple[str, ...] = ()
    required_body_joint_types: tuple[str, ...] = ()
    same_body_as: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EntityKind):
            raise TypeError("kind must be an EntityKind")
        object.__setattr__(self, "allowed_types", tuple(self.allowed_types))
        object.__setattr__(
            self,
            "required_body_joint_types",
            tuple(self.required_body_joint_types),
        )
        if not self.role:
            raise ValueError("role cannot be empty")
        if self.min_count < 0:
            raise ValueError("min_count cannot be negative")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count cannot be smaller than min_count")
        if self.kind is not EntityKind.BODY and self.required_body_joint_types:
            raise ValueError("required_body_joint_types only applies to body roles")
        if (
            self.kind
            not in {EntityKind.GEOM, EntityKind.JOINT, EntityKind.ACTUATOR}
            and self.allowed_types
        ):
            raise ValueError(
                "allowed_types only applies to geom, joint, and actuator roles"
            )


@dataclass(frozen=True)
class BindingContract:
    """Machine-readable requirements published by a primitive or strategy."""

    program_kind: ProgramKind
    program_name: str
    roles: tuple[RoleRequirement, ...]
    parameters: tuple[ParameterRequirement, ...] = ()
    invariants: tuple[str, ...] = ()
    relations: tuple[RelationalInvariant, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "invariants", tuple(self.invariants))
        object.__setattr__(self, "relations", tuple(self.relations))
        if self.program_kind not in {"primitive", "strategy"}:
            raise ValueError(f"unsupported program_kind: {self.program_kind!r}")
        if not self.program_name:
            raise ValueError("program_name cannot be empty")
        names = [requirement.role for requirement in self.roles]
        if len(names) != len(set(names)):
            raise ValueError("binding contract role names must be unique")
        role_names = set(names)
        parameter_names = [requirement.name for requirement in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("binding contract parameter names must be unique")
        if any(
            not isinstance(requirement, ParameterRequirement)
            for requirement in self.parameters
        ):
            raise TypeError(
                "parameters must contain ParameterRequirement values"
            )
        if self.program_kind == "strategy" and self.parameters:
            raise ValueError("strategy binding contracts must be role-only")
        for requirement in self.roles:
            if requirement.same_body_as is None:
                continue
            related = self.roles_by_name.get(requirement.same_body_as)
            if (
                related is None
                or related.kind is not EntityKind.BODY
                or related.min_count != 1
                or related.max_count != 1
            ):
                raise ValueError(
                    f"role {requirement.role!r} same_body_as must reference "
                    "a required singular body role"
                )
        if any(not item for item in self.invariants):
            raise ValueError("invariant descriptions cannot be empty")
        for relation in self.relations:
            if not isinstance(relation, RelationalInvariant):
                raise TypeError(
                    "relations must contain RelationalInvariant values"
                )
            missing = {
                relation.left_role,
                relation.right_role,
            } - role_names
            if missing:
                raise ValueError(
                    f"relational invariant references unknown role(s): "
                    f"{sorted(missing)}"
                )

    @property
    def roles_by_name(self) -> dict[str, RoleRequirement]:
        return {requirement.role: requirement for requirement in self.roles}

    @property
    def parameters_by_name(self) -> dict[str, ParameterRequirement]:
        return {requirement.name: requirement for requirement in self.parameters}


@dataclass(frozen=True)
class SceneBinding:
    """Declarative semantic-name binding for one concrete scene instance."""

    program_kind: ProgramKind
    program_name: str
    instance_name: str
    scene_module: str
    goal_text: str
    roles: Mapping[str, RoleValue]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.program_kind not in {"primitive", "strategy"}:
            raise ValueError(f"unsupported program_kind: {self.program_kind!r}")
        for field_name in (
            "program_name",
            "instance_name",
            "scene_module",
            "goal_text",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        normalized_roles: dict[str, RoleValue] = {}
        for role, value in self.roles.items():
            if isinstance(value, str):
                normalized_roles[str(role)] = value
            else:
                normalized_roles[str(role)] = tuple(value)
        object.__setattr__(self, "roles", MappingProxyType(normalized_roles))
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters),
        )

    def role_values(self, role: str) -> tuple[str, ...]:
        value = self.roles.get(role)
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def one(self, role: str) -> str:
        values = self.role_values(role)
        if len(values) != 1:
            raise ValueError(
                f"role {role!r} must contain exactly one name; got {len(values)}"
            )
        return values[0]


@dataclass(frozen=True)
class BindingIssue:
    """One actionable binding validation diagnostic."""

    code: str
    message: str
    role: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": self.message}
        if self.role is not None:
            payload["role"] = self.role
        return payload


@dataclass(frozen=True)
class BindingReport:
    """Separate structural binding validity from program task success."""

    program_kind: ProgramKind
    program_name: str
    instance_name: str
    scene_module: str
    binding_valid: bool
    program_success: bool | None = None
    issues: tuple[BindingIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.program_kind not in {"primitive", "strategy"}:
            raise ValueError(f"unsupported program_kind: {self.program_kind!r}")
        if self.program_success not in {True, False, None}:
            raise TypeError("program_success must be bool or None")
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def with_program_success(self, success: bool | None) -> "BindingReport":
        return replace(self, program_success=success)

    def require_valid(self) -> None:
        if self.binding_valid:
            return
        detail = "; ".join(issue.message for issue in self.issues)
        raise ValueError(f"scene binding is invalid: {detail}")

    def to_dict(self) -> dict[str, Any]:
        return to_json_value({
            "program_kind": self.program_kind,
            "program_name": self.program_name,
            "instance_name": self.instance_name,
            "scene_module": self.scene_module,
            "binding_valid": self.binding_valid,
            "program_success": self.program_success,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
        })


_OBJECT_TYPES = {
    EntityKind.BODY: mujoco.mjtObj.mjOBJ_BODY,
    EntityKind.GEOM: mujoco.mjtObj.mjOBJ_GEOM,
    EntityKind.SITE: mujoco.mjtObj.mjOBJ_SITE,
    EntityKind.JOINT: mujoco.mjtObj.mjOBJ_JOINT,
    EntityKind.ACTUATOR: mujoco.mjtObj.mjOBJ_ACTUATOR,
}


def _freeze_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return _freeze_mapping(value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )


def _normalized_enum_name(value: Enum | str) -> str:
    text = str(getattr(value, "name", value))
    return (
        text.lower()
        .removeprefix("mjgeom_")
        .removeprefix("mjjnt_")
        .removeprefix("mjtrn_")
    )


def _entity_id(
    model: mujoco.MjModel,
    kind: EntityKind,
    name: str,
) -> int:
    return int(mujoco.mj_name2id(model, _OBJECT_TYPES[kind], name))


def _entity_body_id(
    model: mujoco.MjModel,
    kind: EntityKind,
    entity_id: int,
) -> int | None:
    if kind is EntityKind.BODY:
        return entity_id
    if kind is EntityKind.GEOM:
        return int(model.geom_bodyid[entity_id])
    if kind is EntityKind.SITE:
        return int(model.site_bodyid[entity_id])
    if kind is EntityKind.JOINT:
        return int(model.jnt_bodyid[entity_id])
    return None


def _role_type(
    model: mujoco.MjModel,
    kind: EntityKind,
    entity_id: int,
) -> str | None:
    if kind is EntityKind.GEOM:
        return _normalized_enum_name(mujoco.mjtGeom(int(model.geom_type[entity_id])))
    if kind is EntityKind.JOINT:
        return _normalized_enum_name(mujoco.mjtJoint(int(model.jnt_type[entity_id])))
    if kind is EntityKind.ACTUATOR:
        return _normalized_enum_name(
            mujoco.mjtTrn(int(model.actuator_trntype[entity_id]))
        )
    return None


def _body_joint_types(model: mujoco.MjModel, body_id: int) -> set[str]:
    start = int(model.body_jntadr[body_id])
    count = int(model.body_jntnum[body_id])
    return {
        _normalized_enum_name(mujoco.mjtJoint(int(model.jnt_type[joint_id])))
        for joint_id in range(start, start + count)
    }


def _coerce_invariant_issue(value: BindingIssue | str) -> BindingIssue:
    if isinstance(value, BindingIssue):
        return value
    return BindingIssue(code="invariant_failed", message=str(value))


def _is_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _parameter_type_valid(requirement: ParameterRequirement, value: Any) -> bool:
    if requirement.value_type is ParameterType.FLOAT:
        return _is_json_number(value)
    if requirement.value_type is ParameterType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if requirement.value_type is ParameterType.BOOLEAN:
        return isinstance(value, bool)
    if requirement.value_type is ParameterType.STRING:
        return isinstance(value, str)
    if requirement.value_type is ParameterType.VECTOR:
        return (
            isinstance(value, tuple)
            and len(value) == requirement.vector_length
            and all(_is_json_number(item) for item in value)
        )
    return False


def validate_binding_parameters(
    contract: BindingContract,
    binding: SceneBinding,
) -> list[BindingIssue]:
    issues: list[BindingIssue] = []
    requirements = contract.parameters_by_name
    for name in sorted(set(binding.parameters) - set(requirements)):
        issues.append(
            BindingIssue(
                code="undeclared_parameter",
                role=name,
                message=f"binding declares undeclared primitive parameter {name!r}",
            )
        )
    for name, requirement in requirements.items():
        if name not in binding.parameters:
            if requirement.required:
                issues.append(
                    BindingIssue(
                        code="missing_parameter",
                        role=name,
                        message=f"binding is missing required primitive parameter {name!r}",
                    )
                )
            continue
        value = binding.parameters[name]
        if not _parameter_type_valid(requirement, value):
            issues.append(
                BindingIssue(
                    code="wrong_parameter_type",
                    role=name,
                    message=(
                        f"primitive parameter {name!r} must be "
                        f"{requirement.value_type.value}"
                    ),
                )
            )
            continue
        values = value if isinstance(value, tuple) else (value,)
        numeric = [float(item) for item in values if _is_json_number(item)]
        if requirement.minimum is not None and any(
            item < requirement.minimum for item in numeric
        ):
            issues.append(
                BindingIssue(
                    code="parameter_out_of_range",
                    role=name,
                    message=(
                        f"primitive parameter {name!r} must be at least "
                        f"{requirement.minimum}"
                    ),
                )
            )
        if requirement.maximum is not None and any(
            item > requirement.maximum for item in numeric
        ):
            issues.append(
                BindingIssue(
                    code="parameter_out_of_range",
                    role=name,
                    message=(
                        f"primitive parameter {name!r} must be at most "
                        f"{requirement.maximum}"
                    ),
                )
            )
        if requirement.choices and value not in requirement.choices:
            issues.append(
                BindingIssue(
                    code="parameter_not_allowed",
                    role=name,
                    message=(
                        f"primitive parameter {name!r} must be one of "
                        f"{list(requirement.choices)}"
                    ),
                )
            )
    return issues


def _body_for_entity(
    model: mujoco.MjModel,
    requirement: RoleRequirement,
    entity_id: int,
) -> int | None:
    body_id = _entity_body_id(model, requirement.kind, entity_id)
    if body_id is not None:
        return body_id
    if requirement.kind is not EntityKind.ACTUATOR:
        return None
    transmission = mujoco.mjtTrn(int(model.actuator_trntype[entity_id]))
    if transmission not in {
        mujoco.mjtTrn.mjTRN_JOINT,
        mujoco.mjtTrn.mjTRN_JOINTINPARENT,
    }:
        return None
    joint_id = int(model.actuator_trnid[entity_id, 0])
    if joint_id < 0:
        return None
    return int(model.jnt_bodyid[joint_id])


def _is_descendant_body(
    model: mujoco.MjModel,
    child_body_id: int,
    ancestor_body_id: int,
) -> bool:
    current = child_body_id
    while current != 0:
        current = int(model.body_parentid[current])
        if current == ancestor_body_id:
            return True
    return False


def _validate_relation(
    model: mujoco.MjModel,
    relation: RelationalInvariant,
    requirements: Mapping[str, RoleRequirement],
    resolved: Mapping[str, list[int]],
) -> BindingIssue | None:
    left = resolved.get(relation.left_role, [])
    right = resolved.get(relation.right_role, [])
    if not left or not right:
        return None
    if relation.relation is RelationKind.DISTINCT_ENTITIES:
        valid = (
            requirements[relation.left_role].kind
            is not requirements[relation.right_role].kind
            or not (set(left) & set(right))
        )
    else:
        left_req = requirements[relation.left_role]
        right_req = requirements[relation.right_role]
        left_bodies = {
            body_id
            for entity_id in left
            if (body_id := _body_for_entity(model, left_req, entity_id)) is not None
        }
        right_bodies = {
            body_id
            for entity_id in right
            if (body_id := _body_for_entity(model, right_req, entity_id)) is not None
        }
        if not left_bodies or not right_bodies:
            return BindingIssue(
                code="relation_not_applicable",
                message=(
                    f"cannot resolve bodies for relational invariant: "
                    f"{relation.description}"
                ),
            )
        if relation.relation is RelationKind.SAME_BODY:
            valid = len(left_bodies | right_bodies) == 1
        elif relation.relation is RelationKind.DIFFERENT_BODIES:
            valid = left_bodies.isdisjoint(right_bodies)
        elif relation.relation is RelationKind.DESCENDANT_BODY_OF:
            valid = all(
                any(_is_descendant_body(model, child, ancestor) for ancestor in right_bodies)
                for child in left_bodies
            )
        else:  # pragma: no cover - exhaustive guard for future enum members
            valid = False
    if valid:
        return None
    return BindingIssue(
        code="relational_invariant_failed",
        message=relation.description,
        role=relation.left_role,
    )


def structural_role_candidates(
    model: mujoco.MjModel,
    contract: BindingContract,
) -> dict[str, list[str]]:
    """Filter named entities by unary constraints and singular role relations.

    These are structural candidates, not semantic assignments. Arc consistency
    removes impossible pairs without selecting among equivalent candidates.
    """
    domains: dict[str, list[int]] = {}
    requirements = contract.roles_by_name
    for role, requirement in requirements.items():
        count = {
            EntityKind.BODY: model.nbody,
            EntityKind.GEOM: model.ngeom,
            EntityKind.SITE: model.nsite,
            EntityKind.JOINT: model.njnt,
            EntityKind.ACTUATOR: model.nu,
        }[requirement.kind]
        allowed = {_normalized_enum_name(x) for x in requirement.allowed_types}
        joints = {
            _normalized_enum_name(x) for x in requirement.required_body_joint_types
        }
        domains[role] = [
            index
            for index in range(count)
            if mujoco.mj_id2name(model, _OBJECT_TYPES[requirement.kind], index)
            and (not allowed or _role_type(model, requirement.kind, index) in allowed)
            and (not joints or joints <= _body_joint_types(model, index))
        ]
    relations = list(contract.relations)
    relations.extend(
        RelationalInvariant(
            relation=RelationKind.SAME_BODY,
            left_role=req.role,
            right_role=req.same_body_as,
            description="Direct entity ownership",
        )
        for req in contract.roles
        if req.same_body_as is not None
    )
    relations = [
        relation
        for relation in relations
        if all(
            requirements[role].min_count == requirements[role].max_count == 1
            for role in (relation.left_role, relation.right_role)
        )
    ]
    changed = True
    while changed:
        changed = False
        for relation in relations:
            left, right = relation.left_role, relation.right_role
            pairs = [
                (a, b)
                for a in domains[left]
                for b in domains[right]
                if _validate_relation(
                    model, relation, requirements, {left: [a], right: [b]}
                ) is None
            ]
            supported_domains = (
                (left, {a for a, _ in pairs}), (right, {b for _, b in pairs})
            )
            for role, supported in supported_domains:
                reduced = [index for index in domains[role] if index in supported]
                changed |= reduced != domains[role]
                domains[role] = reduced
    return {
        role: [
            mujoco.mj_id2name(model, _OBJECT_TYPES[requirements[role].kind], i)
            for i in indices
        ]
        for role, indices in domains.items()
    }


def resolve_binding_ownership(
    model: mujoco.MjModel,
    contract: BindingContract,
    binding: SceneBinding,
) -> SceneBinding:
    """Resolve body roles uniquely implied by already selected owned entities.

    Preserve ambiguous or conflicting selections for agent correction. Accept
    a derived mapping only if the complete structural contract validates.
    """
    candidates = structural_role_candidates(model, contract)
    owners: dict[str, set[str]] = {}
    for requirement in contract.roles:
        if requirement.same_body_as is None:
            continue
        values = binding.role_values(requirement.role)
        if not values or any(name not in candidates[requirement.role] for name in values):
            continue
        for name in values:
            index = _entity_id(model, requirement.kind, name)
            owner = _body_for_entity(model, requirement, index)
            if owner is not None:
                owner_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, owner)
                if owner_name:
                    owners.setdefault(requirement.same_body_as, set()).add(owner_name)
    roles = dict(binding.roles)
    for role, names in owners.items():
        if len(names) == 1:
            roles[role] = next(iter(names))
    proposed = replace(binding, roles=roles)
    if validate_scene_binding(model, contract, proposed).binding_valid:
        return proposed
    return binding


def validate_scene_binding(
    model: mujoco.MjModel,
    contract: BindingContract,
    binding: SceneBinding,
    *,
    invariant_checks: Sequence[InvariantCheck] = (),
) -> BindingReport:
    """Resolve and structurally validate one binding against a MuJoCo model."""

    issues: list[BindingIssue] = []
    if binding.program_kind != contract.program_kind:
        issues.append(
            BindingIssue(
                code="program_kind_mismatch",
                message=(
                    f"binding targets {binding.program_kind!r}, contract requires "
                    f"{contract.program_kind!r}"
                ),
            )
        )
    if binding.program_name != contract.program_name:
        issues.append(
            BindingIssue(
                code="program_name_mismatch",
                message=(
                    f"binding targets {binding.program_name!r}, contract requires "
                    f"{contract.program_name!r}"
                ),
            )
        )

    issues.extend(validate_binding_parameters(contract, binding))

    requirements = contract.roles_by_name
    unknown_roles = sorted(set(binding.roles) - set(requirements))
    for role in unknown_roles:
        issues.append(
            BindingIssue(
                code="unknown_role",
                role=role,
                message=f"binding declares unknown role {role!r}",
            )
        )

    resolved: dict[str, list[int]] = {}
    for role, requirement in requirements.items():
        values = binding.role_values(role)
        if len(values) < requirement.min_count:
            issues.append(
                BindingIssue(
                    code="role_count",
                    role=role,
                    message=(
                        f"role {role!r} requires at least {requirement.min_count} "
                        f"name(s); got {len(values)}"
                    ),
                )
            )
        if requirement.max_count is not None and len(values) > requirement.max_count:
            issues.append(
                BindingIssue(
                    code="role_count",
                    role=role,
                    message=(
                        f"role {role!r} permits at most {requirement.max_count} "
                        f"name(s); got {len(values)}"
                    ),
                )
            )

        ids: list[int] = []
        for name in values:
            entity_id = _entity_id(model, requirement.kind, name)
            if entity_id < 0:
                issues.append(
                    BindingIssue(
                        code="missing_entity",
                        role=role,
                        message=(
                            f"{requirement.kind.value} {name!r} for role "
                            f"{role!r} does not exist"
                        ),
                    )
                )
                continue
            ids.append(entity_id)

            actual_type = _role_type(model, requirement.kind, entity_id)
            allowed = {_normalized_enum_name(item) for item in requirement.allowed_types}
            if allowed and actual_type not in allowed:
                issues.append(
                    BindingIssue(
                        code="wrong_entity_type",
                        role=role,
                        message=(
                            f"role {role!r} binds {name!r} with type "
                            f"{actual_type!r}; expected one of {sorted(allowed)}"
                        ),
                    )
                )

            if requirement.kind is EntityKind.BODY:
                actual_joint_types = _body_joint_types(model, entity_id)
                required_joint_types = {
                    _normalized_enum_name(item)
                    for item in requirement.required_body_joint_types
                }
                missing = sorted(required_joint_types - actual_joint_types)
                if missing:
                    issues.append(
                        BindingIssue(
                            code="missing_body_joint_type",
                            role=role,
                            message=(
                                f"body {name!r} for role {role!r} is missing "
                                f"joint type(s) {missing}"
                            ),
                        )
                    )
        resolved[role] = ids

    for role, requirement in requirements.items():
        if requirement.same_body_as is None:
            continue
        target_ids = resolved.get(requirement.same_body_as, [])
        source_ids = resolved.get(role, [])
        target_requirement = requirements.get(requirement.same_body_as)
        if (
            target_requirement is None
            or len(target_ids) != 1
            or target_requirement.kind is not EntityKind.BODY
        ):
            issues.append(
                BindingIssue(
                    code="invalid_same_body_contract",
                    role=role,
                    message=(
                        f"role {role!r} references same_body_as "
                        f"{requirement.same_body_as!r}, which must be a singular "
                        "body role"
                    ),
                )
            )
            continue
        expected_body_id = target_ids[0]
        for entity_id in source_ids:
            actual_body_id = _body_for_entity(model, requirement, entity_id)
            if actual_body_id is not None and actual_body_id != expected_body_id:
                issues.append(
                    BindingIssue(
                        code="wrong_parent_body",
                        role=role,
                        message=(
                            f"role {role!r} is not attached to body role "
                            f"{requirement.same_body_as!r}"
                        ),
                    )
                )

    for relation in contract.relations:
        issue = _validate_relation(model, relation, requirements, resolved)
        if issue is not None:
            issues.append(issue)

    for check in invariant_checks:
        try:
            issues.extend(_coerce_invariant_issue(item) for item in check(model, binding))
        except Exception as exc:
            issues.append(
                BindingIssue(
                    code="invariant_exception",
                    message=f"invariant check {check.__name__} failed: {exc}",
                )
            )

    return BindingReport(
        program_kind=binding.program_kind,
        program_name=binding.program_name,
        instance_name=binding.instance_name,
        scene_module=binding.scene_module,
        binding_valid=not issues,
        issues=tuple(issues),
    )


__all__ = [
    "BindingContract",
    "BindingIssue",
    "BindingReport",
    "EntityKind",
    "InvariantCheck",
    "ParameterRequirement",
    "ParameterType",
    "ProgramKind",
    "RelationKind",
    "RelationalInvariant",
    "RoleRequirement",
    "RoleValue",
    "SceneBinding",
    "resolve_binding_ownership",
    "structural_role_candidates",
    "validate_binding_parameters",
    "validate_scene_binding",
]
