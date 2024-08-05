# Master Thesis
## Hydrodynamical Simulations of Eccentric Mass Transfer

Michael Stroet

Supervisor: Dr. Silvia Toonen

Daily supervisor: Caspar Bruenech MSc

Animations available on YouTube: https://www.youtube.com/playlist?list=PLD-IpZk3HomJ3ju3GYBxuoq9Ehgm2W7W2


### Abstract

_Aims._ Observations have shown that stars often come in systems of two or more stars.
One of the most important stellar interaction mechanics is mass transfer, which has
an enormous impact on the evolutionary outcome of the system and is the origin of
various observed objects and phenomena such as X-ray binaries, ms pulsars, type Ia
supernovae and gravitational waves through mergers. While mass transfer has been
researched for decades, it mostly focused on binaries in circular orbits. However, in
triple systems it is possible for mass transfer to occur on eccentric orbits. As we
still lack a solid theoretical framework for the modelling of eccentric mass transfer,
I performed state-of-the-art hydrodynamical simulations to lay the groundwork for
new models.

_Methods._ I have written a code, based on de Vries et al. (2014) and Parkosidis (2023), which
simulates mass transfer between two stars by combining Smoothed-Particle Hydro-
dynamics (SPH) through Gadget-2 with N-body dynamics through Huayno and
stellar evolution in Mesa, all working together in Amuse. I ran several mass trans-
fer simulations of a 1.5+1.4 M<sub>⊙</sub> binary system with different initial eccentricities,
including circular, and compared their outcomes.

_Results._ Mass transfer in circular and eccentric orbits show stark differences. In
circular orbits, the mass transfer is continuous, while it only occurs around periapsis
in eccentric orbits. These ’_episodes_’ of mass transfer are not centred on periapsis
as expected, but are instead slightly delayed by a constant fraction of the period
between 0.12-0.16. Additionally, systems with higher eccentricities appear to have
increased mass transfer rates and shorter mass transfer windows.

_Conclusions._ Using my code, I have demonstrated several key differences between
circular and eccentric mass transfer, and some best-practices for using SPH in this
case. Several of my findings corroborate the results of previous hydrodynamical
simulations, while also refuting some key assumptions made in current analytical
models. My code can be used in the future to run many more simulations of systems
with different mass ratios and higher eccentricities at larger resolutions.
