import heapq

class Node:
    def __init__(self, freq, symbol=None, children=None):
        self.freq = freq
        self.symbol = symbol
        self.children = children if children else []

    def __lt__(self, other):
        return self.freq < other.freq

def build_4ary_huffman(frequencies):
    n = len(frequencies)
    remainder = (n - 1) % 3
    if remainder != 0:
        for i in range(3 - remainder):
            frequencies.append(Node(0, symbol=f"dummy_{i}"))

    heap = frequencies.copy()
    heapq.heapify(heap)

    while len(heap) > 1:
        nodes = []
        for _ in range(4):
            if heap:
                nodes.append(heapq.heappop(heap))

        # Создаем родительский узел
        total_freq = sum(node.freq for node in nodes)
        parent = Node(total_freq, children=nodes)
        heapq.heappush(heap, parent)

    return heap[0] if heap else None

def assign_codes(root):
    codes = {}
    def traverse(node, code):
        if node.symbol:
            if not node.symbol.startswith("dummy"):
                codes[node.symbol] = code
        else:
            for i, child in enumerate(node.children):
                traverse(child, code + str(i))
    traverse(root, "")
    return codes

if __name__ == "__main__":
    symbols = [
        Node(16.2, "|"), Node(10.8, "O"), Node(6.42, "И"), Node(6.28, "E"),
        Node(5.55, "Л"), Node(5.4, "A"), Node(5.4, "Н"), Node(4.38, "K"),
        Node(3.94, "T"), Node(3.65, "B"), Node(3.36, "C"), Node(3.21, "Д"),
        Node(3.07, "P"), Node(2.34, "М"), Node(2.19, "Y"), Node(2.19, "bl"),
        Node(2.19, "b"), Node(2.04, "Г"), Node(2.04, "П"), Node(1.31, "Я"),
        Node(1.17, "З"), Node(1.17, "X"), Node(1.02, "Й"), Node(0.88, "Ш"),
        Node(0.73, "Ж"), Node(0.58, "Б"), Node(0.58, "Ц"), Node(0.58, "Ч"),
        Node(0.58, "Щ"), Node(0.44, "Ю"), Node(0.29, "Ъ"), Node(0, "Ф"), Node(0, "Э")
    ]

    # Построение дерева
    root = build_4ary_huffman(symbols)

    huffman_codes = assign_codes(root)

    for symbol, code in sorted(huffman_codes.items(), key=lambda x: (-len(x[1]), x[1])):
        print(f"{symbol}: {code}")