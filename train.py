#!/usr/bin/env python
import main

if __name__ == "__main__":
    main.train()
    main.evaluate()
    import os, signal
    os.kill(os.getpid(), signal.SIGKILL)