import os

title = input("setting title:")
w = input("setting width:")
h = input("setting height:")
mapid = input("selecting map")
debug = input("show hitbox? (y/n)")

if debug == "y":
    debug = "on"

os.system("RTRA -title " + title + " -size " + w + " " + h + " -map " + mapid + " -hitbox " + debug)