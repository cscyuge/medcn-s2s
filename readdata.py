import pickle
import os


def read_data():
    data_path = "./data/data/"
    src = []
    tar = []
    done = set()
    for root, dirs, files in os.walk(data_path):
        if len(dirs) > 0:
            for dir in dirs:
                print(dir)
                if dir=="admin":
                    continue
                for _, __, ___ in os.walk(os.path.join(data_path,dir)):
                    print(len(___))
                    for file in ___:
                        if file[0] in '0123456789c':
                            with open(os.path.join(data_path,dir,file),'rb') as f:
                                data = pickle.load(f)
                                if data['textid'] not in done:
                                    done.add(data['textid'])
                                    src.append(data['src'])
                                    tar.append(data['tar'])
    return src, tar



def main():
    src_all, tar_all = read_data()
    with open('./data/src_all.pkl','wb') as f:
        pickle.dump(src_all, f)
    with open('./data/tar_all.pkl', 'wb') as f:
        pickle.dump(tar_all, f)


if __name__ == '__main__':
    main()