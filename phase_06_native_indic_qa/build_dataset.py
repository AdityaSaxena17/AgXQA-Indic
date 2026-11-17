"""Implementation starter for phase 6: build_dataset. See README.md."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='phase_06_native_indic_qa/config.json')
    parser.parse_args()
    parser.exit(2, 'Pending implementation: build_dataset. Follow the phase README and record real artifacts.\n')

if __name__ == '__main__':
    main()
