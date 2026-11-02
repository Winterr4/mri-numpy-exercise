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

### FASE 1: generar cubo de datos ---------------------------------------------------------------------
import numpy as np
np.random.seed(44)
mri_i = np.random.normal(loc=120, scale=10, size=(100, 100, 100))
print(mri_i.shape)

### FASE 2: calcular media de todo el eje z(numeros de dentro (xy) de cada z, osea [:,:,z]=media)-----------------
media_i=np.mean(mri_i, axis=(0,1))


### FASE 3: restar esa media a cada xy con broadcasting (necesito reshape)
media_i_3D= media_i[np.newaxis, np.newaxis, :]
#ó: media_i_3D= media_i.reshape((1, 1, 100)) 

mri_c= mri_i - media_i_3D

### FASE 4: evaluar si ahora xy (medias) es ==0, y si la std se mantiene cosntante

media_c= np.mean(mri_c,axis=(0,1))
print(np.allclose(media_c, 0))
print(mri_i[:1,:1,:1])
print(mri_c[:1,:1,:1])

print(mri_i.std())
print(mri_c.std())
print(np.allclose(mri_i.std(), mri_c.std()))

diff = abs(mri_i.std() - mri_c.std())
print(diff)