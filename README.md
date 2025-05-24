## astro-batch
# Overview
Vectorised functions for common astrodynamics calculations. Useful for dealing with large numbers of orbits.
If you only have small sets of orbits that need to be processed in one go (perhaps up to 10-50), you're much better off using ESA's pykep library: https://esa.github.io/pykep/

This started off as a tool to help my research, where I train neural networks and thus often need to process large datasets of orbits quickly. Pykep's functions are very fast but many of them do not yet support vectorised calculations so you end up needing to do a loop, which obviously gets slow for large numbers of orbits.

Making vectorised versions of the common conversions is not difficult but I ended up redoing it every time I start a new project, so decided to make a more rigorous implementation that I can turn to whenever I need it in the future. There's nothing scientifically original here, it's just a basic tool so I thought I might as well release it to the public in case it saves someone else time.

I'm likely to keep updating this when I have the time.

# Limitations
Currently only designed to work with elliptical prograde orbits.
