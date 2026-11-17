"""Print a planning config; never execute an experiment."""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps({'mode': 'plan_only', 'experiment_executed': False, 'config': config}, indent=2))

if __name__ == '__main__':
    main()
