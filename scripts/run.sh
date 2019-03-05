python3 simulation.py
python3 -m dedalus merge_procs snapshots --cleanup
python3 -m dedalus merge_procs coefficients --cleanup
rm -rf frames
rm -rf frames_coeffs
python3 plot_profiles.py snapshots/*.h5
python3 plot_coeffs.py coefficients/*.h5
