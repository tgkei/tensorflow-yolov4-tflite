import json
from pprint import pprint

TIME_GAP = 300


def main(json_file):
    obj = None
    with open(json_file, "r") as f:
        obj = json.load(f)
    obj = json.loads(obj)
    new_dict = dict()
    for class_name, frames in obj.items():
        tmp = []
        start = frames[0]
        prev = start
        cur = start
        end = start
        idx = 1
        while idx < len(frames):
            cur = frames[idx]
            if abs(cur - prev) < TIME_GAP:
                prev = cur
            else:
                end = prev
                tmp.append([start, end])
                start = cur
                prev = start
            idx += 1
        else:
            end = prev
            tmp.append([start, end])

        new_dict[class_name] = tmp

    new_json = json.dumps(new_dict)
    with open("final.json", "w") as f:
        json.dump(new_json, f)


if __name__ == "__main__":
    json_file = "./test.json"
    main(json_file)
