"""Implementation starter for phase 4: build_corpus. See README.md."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='phase_04_indic_aec_corpus/config.json')
    parser.parse_args()
    parser.exit(2, 'Pending implementation: build_corpus. Follow the phase README and record real artifacts.\n')

if __name__ == '__main__':
    main()
