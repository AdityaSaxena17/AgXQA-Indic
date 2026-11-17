"""Implementation starter for phase 3: train. See README.md."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='phase_03_multilingual_finetuning/config.json')
    parser.parse_args()
    parser.exit(2, 'Pending implementation: train. Follow the phase README and record real artifacts.\n')

if __name__ == '__main__':
    main()
