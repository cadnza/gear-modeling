import math
import random
from dataclasses import dataclass
from typing import Any, cast

from sympy import (
    Eq,
    Expr,
    S,
    Symbol,
    solve,  # pyright: ignore[reportUnknownVariableType]
    solveset,  # pyright: ignore[reportUnknownVariableType]
    symbols,  # pyright: ignore[reportUnknownVariableType]
)

MINIMUM_TEETH = 5
"""A working minimum gear tooth count."""


@dataclass(frozen=True, kw_only=True)
class _PlanetaryGearsetVelocity:
    """A planetary gearset velocity."""

    w_sun: float
    """The velocity of the sun gear."""

    w_planets: float
    """The velocity of the planet gears."""

    w_ring: float
    """The velocity of the ring gears."""

    w_carrier: float
    """The velocity of the carrier."""


class PlanetaryGearset:
    """A planetary gear configuration."""

    z_sun: int
    """The number of teeth in the sun gear."""

    z_planet: int
    """The number of teeth in the planet gear."""

    z_ring: int
    """The number of teeth in the ring gear."""

    n_planet_gears: int
    """The number of planet gears."""

    kinematics: PlanetaryKinematics
    """Kinematics."""

    def _calculate_z_sun(self, *, z_planet: int, z_ring: int) -> int:
        return int(
            next(
                iter(
                    solveset(
                        self.kinematics.eq_mesh.subs(  # type: ignore  # noqa: PGH003
                            {
                                self.kinematics.zp: z_planet,
                                self.kinematics.zr: z_ring,
                            },
                        ),
                        self.kinematics.zs,
                        domain=S.Reals,
                    ),
                ),
            ),
        )

    def _calculate_z_planet(self, *, z_sun: int, z_ring: int) -> int:
        return int(
            next(
                iter(
                    solveset(
                        self.kinematics.eq_mesh.subs(  # type: ignore  # noqa: PGH003
                            {self.kinematics.zs: z_sun, self.kinematics.zr: z_ring},
                        ),
                        self.kinematics.zp,
                        domain=S.Reals,
                    ),
                ),
            ),
        )

    def _calculate_z_ring(self, *, z_sun: int, z_planet: int) -> int:
        return int(
            next(
                iter(
                    solveset(
                        self.kinematics.eq_mesh.subs(  # type: ignore  # noqa: PGH003
                            {self.kinematics.zs: z_sun, self.kinematics.zp: z_planet},
                        ),
                        self.kinematics.zr,
                        domain=S.Reals,
                    ),
                ),
            ),
        )

    def __repr__(self) -> str:
        return f"PlanetaryGearset(z_sun = {self.z_sun:3d}, z_planet = {self.z_planet:3d}, z_ring = {self.z_ring:3d}, n_planets = {self.n_planet_gears:3d})"

    def __init__(
        self,
        *,
        z_sun: int | None = None,
        z_planet: int | None = None,
        z_ring: int | None = None,
        n_planet_gears: int,
        kinematics: PlanetaryKinematics,
    ) -> None:
        # Commit symbols
        self.kinematics = kinematics

        # Validate and commit planet gear count
        min_planet_gears = 2
        if n_planet_gears < min_planet_gears:
            msg = f"Please set planet gear count to at least {min_planet_gears}"
            raise TooFewPlanetGearsError(
                msg,
            )
        self.n_planet_gears = n_planet_gears

        # Make sure a minimum number of parameters has been supplied
        n_params = (
            (0 if z_sun is None else 1)
            + (0 if z_planet is None else 1)
            + (0 if z_ring is None else 1)
        )
        min_params = 2
        if n_params < min_params:
            msg = f"Please specify at least {min_params} tooth counts"
            raise TooFewToothCountsSpecifiedError(msg)

        # Validate the case of three parameters
        max_params = 3
        if n_params == max_params and not all(
            [
                z_sun
                == self._calculate_z_sun(z_planet=z_planet or 0, z_ring=z_ring or 0),
                z_planet
                == self._calculate_z_planet(z_sun=z_sun or 0, z_ring=z_ring or 0),
                z_ring
                == self._calculate_z_ring(z_sun=z_sun or 0, z_planet=z_planet or 0),
            ],
        ):
            msg_0 = "Invalid configuration given tooth counts"
            raise InvalidConfigurationGivenToothCountsError(msg_0)

        # Commit gear tooth counts
        self.z_sun = z_sun or self._calculate_z_sun(
            z_planet=z_planet or 0,
            z_ring=z_ring or 0,
        )
        self.z_planet = z_planet or self._calculate_z_planet(
            z_sun=z_sun or 0,
            z_ring=z_ring or 0,
        )
        self.z_ring = z_ring or self._calculate_z_ring(
            z_sun=z_sun or 0,
            z_planet=z_planet or 0,
        )

        # Validate gear tooth counts
        if self.z_sun < MINIMUM_TEETH:
            msg_1 = (
                f"Sun gear tooth count must be greater than or equal to {MINIMUM_TEETH}"
            )
            raise PlanetGearToothCountTooLowError(
                msg_1,
            )
        if self.z_planet < MINIMUM_TEETH:
            msg_1 = f"Planet gear tooth count must be greater than or equal to {MINIMUM_TEETH}"
            raise PlanetGearToothCountTooLowError(
                msg_1,
            )
        if self.z_ring < MINIMUM_TEETH:
            msg_1 = f"Ring gear tooth count must be greater than or equal to {MINIMUM_TEETH}"
            raise PlanetGearToothCountTooLowError(
                msg_1,
            )

        # Make sure planet gears can be evenly spaced
        if not (self.z_sun + self.z_ring / self.n_planet_gears).is_integer():
            msg_1 = "Planet gears cannot be evenly spaced"
            raise PlanetGearsCannotBeEvenlySpacedError(msg_1)

        # Make sure planet gears don't overlap
        if self.z_planet + 2 >= (self.z_sun + self.z_planet) * math.sin(
            math.pi / self.n_planet_gears,
        ):
            msg_2 = "Planet gears are overlapping"
            raise PlanetGearsOverlappingError(msg_2)

    def drive(
        self,
        *,
        w_sun: float | None = None,
        w_planets: float | None = None,
        w_carrier: float | None = None,
        w_ring: float | None = None,
    ) -> _PlanetaryGearsetVelocity:
        """Drive this gearset."""
        # Validate inputs
        provided = {
            "w_sun": w_sun,
            "w_planets": w_planets,
            "w_carrier": w_carrier,
            "w_ring": w_ring,
        }
        given = {k for k, v in provided.items() if v is not None}
        min_given = 2
        if len(given) != min_given:
            msg = (
                f"Invalid drive configuration: provide exactly {min_given} of "
                "w_sun, w_planets, w_carrier, w_ring"
            )
            raise InvalidDriveConfigurationError(
                msg,
            )

        # Case: w_sun and w_carrier are known -> solve for wp, wr
        if given == {"w_sun", "w_carrier"}:
            eq1 = self.kinematics.eq_sun__planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.ws: w_sun,
                    self.kinematics.wc: w_carrier,
                },
            )
            eq2 = self.kinematics.eq_ring_planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wc: w_carrier,
                    # ws isn't used in the second constraint; leaving wr, wp as unknowns
                },
            )

            sol = solve(  # pyright: ignore[reportUnknownVariableType]
                (eq1, eq2),
                (self.kinematics.wp, self.kinematics.wr),
                domain=S.Reals,
            )
            # `solve` returns a dict-like mapping of symbols to expressions
            wp_val = float(sol[self.kinematics.wp])  # pyright: ignore[reportUnknownArgumentType]
            wr_val = float(sol[self.kinematics.wr])  # pyright: ignore[reportUnknownArgumentType]
            return _PlanetaryGearsetVelocity(
                w_sun=float(w_sun),  # pyright: ignore[reportArgumentType]
                w_planets=wp_val,
                w_ring=wr_val,
                w_carrier=float(w_carrier),  # pyright: ignore[reportArgumentType]
            )

        # Case: w_sun and w_ring are known -> solve for wp, wc
        if given == {"w_sun", "w_ring"}:
            eq1 = self.kinematics.eq_sun__planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.ws: w_sun,
                },
            )
            eq2 = self.kinematics.eq_ring_planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wr: w_ring,
                },
            )
            sol = solve(  # pyright: ignore[reportUnknownVariableType]
                (eq1, eq2),
                (self.kinematics.wp, self.kinematics.wc),
                domain=S.Reals,
            )
            wp_val = float(sol[self.kinematics.wp])  # pyright: ignore[reportUnknownArgumentType]
            wc_val = float(sol[self.kinematics.wc])  # pyright: ignore[reportUnknownArgumentType]
            return _PlanetaryGearsetVelocity(
                w_sun=float(w_sun),  # pyright: ignore[reportArgumentType]
                w_planets=wp_val,
                w_ring=float(w_ring),  # pyright: ignore[reportArgumentType]
                w_carrier=wc_val,
            )

        # Case: w_carrier and w_ring are known -> solve for ws, wp
        if given == {"w_carrier", "w_ring"}:
            eq1 = self.kinematics.eq_sun__planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wc: w_carrier,
                },
            )
            eq2 = self.kinematics.eq_ring_planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wc: w_carrier,
                    self.kinematics.wr: w_ring,
                },
            )
            sol = solve(  # pyright: ignore[reportUnknownVariableType]
                (eq1, eq2),
                (self.kinematics.ws, self.kinematics.wp),
                domain=S.Reals,
            )
            ws_val = float(sol[self.kinematics.ws])  # pyright: ignore[reportUnknownArgumentType]
            wp_val = float(sol[self.kinematics.wp])  # pyright: ignore[reportUnknownArgumentType]
            return _PlanetaryGearsetVelocity(
                w_sun=ws_val,
                w_planets=wp_val,
                w_ring=float(w_ring),  # pyright: ignore[reportArgumentType]
                w_carrier=float(w_carrier),  # pyright: ignore[reportArgumentType]
            )

        # Case: w_sun and w_planets are known -> solve for wc, wr
        if given == {"w_sun", "w_planets"}:
            eq1 = self.kinematics.eq_sun__planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.ws: w_sun,
                    self.kinematics.wp: w_planets,
                },
            )
            eq2 = self.kinematics.eq_ring_planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wp: w_planets,
                },
            )
            sol = solve(  # pyright: ignore[reportUnknownVariableType]
                (eq1, eq2),
                (self.kinematics.wc, self.kinematics.wr),
                domain=S.Reals,
            )
            wc_val = float(sol[self.kinematics.wc])  # pyright: ignore[reportUnknownArgumentType]
            wr_val = float(sol[self.kinematics.wr])  # pyright: ignore[reportUnknownArgumentType]
            return _PlanetaryGearsetVelocity(
                w_sun=float(w_sun),  # pyright: ignore[reportArgumentType]
                w_planets=float(w_planets),  # pyright: ignore[reportArgumentType]
                w_ring=wr_val,
                w_carrier=wc_val,
            )

        # Case: w_planets and w_ring are known -> solve for ws, wc
        if given == {"w_planets", "w_ring"}:
            eq1 = self.kinematics.eq_sun__planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wp: w_planets,
                },
            )
            eq2 = self.kinematics.eq_ring_planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wp: w_planets,
                    self.kinematics.wr: w_ring,
                },
            )
            sol = solve(  # pyright: ignore[reportUnknownVariableType]
                (eq1, eq2),
                (self.kinematics.ws, self.kinematics.wc),
                domain=S.Reals,
            )
            ws_val = float(sol[self.kinematics.ws])  # pyright: ignore[reportUnknownArgumentType]
            wc_val = float(sol[self.kinematics.wc])  # pyright: ignore[reportUnknownArgumentType]
            return _PlanetaryGearsetVelocity(
                w_sun=ws_val,
                w_planets=float(w_planets),  # pyright: ignore[reportArgumentType]
                w_ring=float(w_ring),  # pyright: ignore[reportArgumentType]
                w_carrier=wc_val,
            )

        # Case: w_planets and w_carrier are known -> solve for wr, ws
        if given == {"w_planets", "w_carrier"}:
            eq1 = self.kinematics.eq_sun__planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wp: w_planets,
                    self.kinematics.wc: w_carrier,
                },
            )
            eq2 = self.kinematics.eq_ring_planet_carrier.subs(  # pyright: ignore[reportUnknownMemberType]
                {
                    self.kinematics.zs: self.z_sun,
                    self.kinematics.zp: self.z_planet,
                    self.kinematics.zr: self.z_ring,
                    self.kinematics.wp: w_planets,
                    self.kinematics.wc: w_carrier,
                },
            )
            sol = solve(  # pyright: ignore[reportUnknownVariableType]
                (eq1, eq2),
                (self.kinematics.wr, self.kinematics.ws),
                domain=S.Reals,
            )
            wr_val = float(sol[self.kinematics.wr])  # pyright: ignore[reportUnknownArgumentType]
            ws_val = float(sol[self.kinematics.ws])  # pyright: ignore[reportUnknownArgumentType]
            return _PlanetaryGearsetVelocity(
                w_sun=ws_val,
                w_planets=float(w_planets),  # pyright: ignore[reportArgumentType]
                w_ring=wr_val,  # pyright: ignore[reportArgumentType]
                w_carrier=float(w_carrier),  # pyright: ignore[reportArgumentType]
            )

        # Raise bad configuration
        msg = "Invalid drive configuration"
        raise InvalidDriveConfigurationError(msg)

    @property
    def distance_sun_to_planet(self) -> float:
        """The distance from the center of the sun gear to the center ofa planet gear."""
        return (self.z_sun + self.z_planet) / 2


class PlanetaryKinematics:
    """A description of planetary gear kinematics."""

    zs: Any
    """The number of teeth on the sun gear."""

    zp: Any
    """The number of teeth on each planet gear."""

    zr: Any
    """The number of teeth on the ring gear."""

    ws: Any
    """The angular velocity of the sun gear with respect to the inertial frame."""

    wp: Any
    """The angular velocity of each planet gear with respect to the inertial frame."""

    wr: Any
    """The angular velocity of the ring gear with respect to the inertial frame."""

    wc: Any
    """The angular velocity of the planet carrier with respect to the inertial frame."""

    def __init__(self) -> None:
        designator = random.randint(1, 10000)  # noqa: S311
        self.zs, self.zp, self.zr = symbols(
            " ".join([f"{x}{designator}" for x in ["zs", "zp", "zr"]]),
            integer=True,
            positive=True,
        )
        self.ws, self.wp, self.wr, self.wc = symbols(
            " ".join([f"{x}{designator}" for x in ["ws", "wp", "wr", "wc"]]),
        )

    @property
    def eq_mesh(self) -> Eq:
        """The gear mesh equation."""
        return cast(
            "Eq",
            Eq(
                self.zr,
                self.zs + 2 * self.zp,
            ),
        )

    @property
    def constraint_mesh(self) -> tuple[Symbol, Expr]:
        """The gear mesh constraint."""
        return (
            cast("Symbol", self.eq_mesh.args[0]),
            cast("Expr", self.eq_mesh.args[1]),
        )

    @property
    def eq_sun__planet_carrier(self) -> Eq:
        """A kinematic equation."""
        return cast(
            "Eq",
            Eq(
                self.zs * self.ws + self.zp * self.wp - (self.zs + self.zp) * self.wc,
                0,
            ),
        )

    @property
    def eq_ring_planet_carrier(self) -> Eq:
        """A kinematic constraint."""
        return cast(
            "Eq",
            Eq(
                self.zr * self.wr - self.zp * self.wp - (self.zr - self.zp) * self.wc,
                0,
            ),
        )

    def gearset(
        self,
        *,
        z_sun: int | None = None,
        z_planet: int | None = None,
        z_ring: int | None = None,
        n_planet_gears: int,
    ) -> PlanetaryGearset:
        """Create a gearset from these kinematics."""
        return PlanetaryGearset(
            z_sun=z_sun,
            z_planet=z_planet,
            z_ring=z_ring,
            n_planet_gears=n_planet_gears,
            kinematics=self,
        )


class TooFewPlanetGearsError(Exception):
    pass


class TooFewToothCountsSpecifiedError(Exception):
    pass


class InvalidConfigurationGivenToothCountsError(Exception):
    pass


class PlanetGearToothCountTooLowError(Exception):
    pass


class PlanetGearsCannotBeEvenlySpacedError(Exception):
    pass


class PlanetGearsOverlappingError(Exception):
    pass


class InvalidDriveConfigurationError(Exception):
    pass
