To install any of these drivers, run:
sudo service gdm3 stop
sudo sh ./<driver_runfile_name_here>.run --silent
If you get any errors, run nvidia-smi to make sure nothing else is running.
If you run into any trouble, email wagle@cs.unc.edu

After changing the driver, also ensure the CUDA Runtime version is the desired version. This can be done by changing the related paths in Makefile to the tested CUDA runtime version (available are cuda-12.2.2 and cuda-11.2) and then perform the experiments following the README.md located inside the testing folders.

After you are done with CUDA test section, make sure you have reinstated the 550 driver.

