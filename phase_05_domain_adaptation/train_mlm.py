"""Implementation starter for phase 5: train_mlm. See README.md."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='phase_05_domain_adaptation/config.json')
    parser.parse_args()
    parser.exit(2, 'Pending implementation: train_mlm. Follow the phase README and record real artifacts.\n')

if __name__ == '__main__':
    main()
