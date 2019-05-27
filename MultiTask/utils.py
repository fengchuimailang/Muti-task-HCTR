

def read_alphabet(filename):
    """
    读取字典返回idx2symbol, symbol2idx
    :param filename:
    :return:
    """
    # TODO 字母表改成自己的版本
    file = []
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            raw = f.readline()
            if not raw:
                break
            file.append(raw)
    idx2symbol = [s.strip('\n') for s in file]
    for i in range(92):
        idx2symbol[i] = idx2symbol[i][1:]
    idx2symbol.insert(0, '<pad>')  # 空白
    # idx2symbol.insert(1, '<GO>')  # 没有作用
    # idx2symbol.insert(2, '<EOS>')  # 结束
    print('alphabet len:', len(idx2symbol))
    symbol2idx = {}
    for idx, symbol in enumerate(idx2symbol):
        symbol2idx[symbol] = idx
    return idx2symbol, symbol2idx