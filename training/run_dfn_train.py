"""Run upstream training with exceptions propagated rather than logger-swallowed."""
import inspect
from df import train
from icecream import ic, install

if __name__ == '__main__':
    ic.includeContext = True
    install()
    entry = inspect.unwrap(train.main)
    assert entry is not train.main, 'Expected pinned upstream logger.catch wrapper'
    try:
        entry()
    except SystemExit as error:
        if error.code in (None, 0) and train.should_stop:
            raise SystemExit(2)  # Partial run is resumable, not completed.
        raise
    print('DFN TRAINER COMPLETED')
