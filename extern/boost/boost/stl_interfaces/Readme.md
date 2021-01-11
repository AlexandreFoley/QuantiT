This is a boost library from boost 1.74 (august 2020 release).
At the time of writing this, that version of boost is not available on debian 
stable's repository, and many other distribution i suspect.

This component of boost has minimal dependencies on the rest of boost, exept for
the testing, which was excised.

All other dependencies were located in sequence_container_interface.hpp and 
were on the BOOST_ASSERT macro. c assert were drop-in replacement for the usage 
made.
