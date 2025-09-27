import subprocess, sys
subprocess.run([sys.executable, "/mnt/data/Assignment7_Package/cpa_assignment7.py",
                "--traces", "/mnt/data/Traces00000.dat",
                "--plaintexts", "/mnt/data/plaintexts.dat",
                "--ciphertexts", "/mnt/data/ciphertexts.dat",
                "--outdir", "/mnt/data/Assignment7_Out",
                "--lastname", "OmidiZadeh",
                "--model", "sbox_hw",
                "--roi", "800",
                "--steps", "20"], check=True)
