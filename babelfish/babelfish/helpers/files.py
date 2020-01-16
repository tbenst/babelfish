from glob import glob

def glob_one_file(path):
    results = glob(path)
    if len(results) == 1:
        return results[0]
    else:
        print("found " + str(len(results)) + " files instead of one for " + path)
        return ""