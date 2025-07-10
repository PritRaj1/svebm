import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

requirements = [
    "torch",
    "torchvision", 
    "torchaudio",
    "lightning[pytorch-extra]",
    "matplotlib",
    "seaborn",
    "numpy",
    "pandas",
    "pytest",
]

def main():
    print("Installing requirements...")

    for package in requirements:
        print(f"Installing {package}...")

        try:
            install(package)
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            sys.exit(1)

    print("Requirements installed successfully!")

def test_install():
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}")
            x = torch.rand(3, 3).cuda()
            y = torch.rand(3, 3).cuda()
            z = x + y
            print("GPU test successful. Tensor sum:\n", z)

        else:
            print("CUDA is NOT available. Torch is using CPU.")
            x = torch.rand(3, 3)
            y = torch.rand(3, 3)
            z = x + y
            print("CPU test successful. Tensor sum:\n", z)

    except ImportError:
        print("torch is not installed.")
        return False
    
    except Exception as e:
        print(f"An error occurred while testing torch GPU: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
    print("\n== Testing installation ==")
    test_install()