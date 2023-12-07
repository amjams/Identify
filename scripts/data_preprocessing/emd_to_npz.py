# imports
import numpy as np
import hyperspy.api as hs
import os



frames = [20]
folder = ''   # parent dicrectory with the strcuture below:

# folder
# |   EMD   (this contains the EMD files)
# |   NPZ   (this is empty intitally)


file_names = []
for file in os.listdir(os.path.join(folder, 'EMD')):
    if file.endswith('.emd'):
        file_names.append(file)

print(file_names)

for last_frame in frames:
    for file_name in file_names:
        # load new file and save to numpy
        file_path = os.path.join(folder, 'EMD', file_name)
        print(file_path)
        s = hs.load(file_path, SI_dtype='uint8', first_frame=1,
                    last_frame=last_frame, sum_frames=True, select_type=None)

        # for i in range(len(s)):
        # 	print(i,s[i])
        # break

        # search for the right data
        for i in range(len(s)):
            if '(2048, 2048|4096)' in repr(s[i]):
                spectrum_idx = i
            elif 'HAADF' in repr(s[i]):
                haadf_idx = i

        haadf = s[haadf_idx].data
        spectrum = s[spectrum_idx].data
        xray_energies = s[spectrum_idx].axes_manager.signal_axes[0].axis

        out_path = os.path.join(
            folder, 'NPZ', file_name[:-4]) + '_%02d.npz' % last_frame
        print(out_path)
        np.savez_compressed(out_path, haadf=haadf,
                        spectrum=spectrum, xray_energies=xray_energies)
        print("Saved for %02d frames" % last_frame)
        del haadf, spectrum, s, xray_energies
