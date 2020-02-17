import asyncio
import os

with open('../Dockerfile') as f:
    for line in f:
        if line.startswith("ENV"):
            cmds = line[:-1].split(' ')
            print(cmds)
            if cmds[0] == "ENV":
                key = cmds[1]
                if cmds[2].startswith('"') or cmds[2].startswith("'"):
                    cmds[2] = cmds[2][1:]
                    cmds[-1] = cmds[-1][:-1]

                value = ' '.join(cmds[2:])
                os.environ[key] = value

from main import main

asyncio.run(main())
