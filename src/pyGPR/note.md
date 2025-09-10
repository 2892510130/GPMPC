## Compare the update method of RLS / Kalman-gain between FITC and DTC
- If the new data is in the region (X or U? Need test) then both method can adapt to the new date,
but if the new data is far from the region, FITC will refuse the update as Kdiag - Qdiag of the 
new data will be close to 1 (Qdia close to 0, if kernel is normalized).
- If the original inducing points are left half of the data space, and we update the right half using 
the algorithm then it is not working.
- If the original inducing points are larger than the data space, for example, data space is left half
of the inducing points (inducing points are not dense, the are less still), and we update the right half,
it is really good!