"""Implementation starter for phase 1: preprocess. See README.md."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='phase_01_original_baseline/config.json')
    parser.parse_args()
    parser.exit(2, 'Pending implementation: preprocess. Follow the phase README and record real artifacts.\n')

if __name__ == '__main__':
    main()
