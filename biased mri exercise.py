'''
* You have a 3D MRI scan of a single subject (100x100x100 voxels). Due to "bias field" effects, different slices (along the Z-axis) have slightly different average intensities. To correct this, you need to calculate the mean intensity for each 2D slice and subtract that mean from every voxel in that specific slice.

* The task:
  * Create a 100x100x100 array of random floats to simulate the MRI volume.
  * Calculate the mean intensity for each of the 100 slices along the Z (third) axis (axis=2).
  * Reshape the resulting means and use Broadcasting to normalize the entire volume.
    * Hint: Reshape the 1D means into a 3D shape that allows NumPy to 'stretch' each mean value across all voxels in its corresponding slice before performing the subtraction.
  * Raise your hand and explain how you checked the correctness of your code.
    * I'm interested in hearing you explain how you know the mean was computed on the right slices, and how the right reshaping/bradcasting was done.  
    * Hint: Since the goal is to subtract the mean from every slice, the new mean of every slice should be (theoretically) zero. And the global Standard deviation should stay the same.
'''


import numpy as np

### STEP 1: generate data cube ---------------------------------------------------------------------
import numpy as np
np.random.seed(44)
mri_i = np.random.normal(loc=120, scale=10, size=(100, 100, 100))
print(mri_i.shape)

## STEP 2: compute the mean of the whole z axis (the numbers inside (xy) for each z, i.e. [:,:,z]=mean)-----------------

media_i=np.mean(mri_i, axis=(0,1))


### STEP 3: subtract that mean from each xy slice with broadcasting (reshape needed)
media_i_3D= media_i[np.newaxis, np.newaxis, :]
#or: media_i_3D= media_i.reshape((1, 1, 100)) 

mri_c= mri_i - media_i_3D

## STEP 4: check whether the means are now ==0, and whether the std stays constant

media_c= np.mean(mri_c,axis=(0,1))
print(np.allclose(media_c, 0))
print(mri_i[:1,:1,:1])
print(mri_c[:1,:1,:1])

print(mri_i.std())
print(mri_c.std())
print(np.allclose(mri_i.std(), mri_c.std()))

diff = abs(mri_i.std() - mri_c.std())
print(diff)

"""
How i checked it:
Bias field correction on a simulated 3D MRI volume using NumPy broadcasting.

How I checked it:
1. Computed the mean with axis=(0,1), which collapses X and Y and leaves
   one mean per Z slice. That's why the result has shape (100,).
2. Reshaped it to (1,1,100) so NumPy aligns dimensions from the right and
   subtracts each mean from its entire slice.
3. Verified that the mean of each slice after correction is ≈ 0 with np.allclose.
4. The global std barely changes (diff ≈ 5e-4). This is expected: removing
   the slice means eliminates the variance *between* slices, which is very
   small compared to the total variance.
"""
