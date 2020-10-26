import xlrd
import numpy as np


def read_data():
    names = ["AI experiments/id3/train_data", "AI experiments/id3/test_data"]
    data = []
    for i in [0, 1]:
        fname = names[i] + ".xlsx"
        bk = xlrd.open_workbook(fname)
        shxrange = range(bk.nsheets)
        try:
            sh = bk.sheet_by_name("Sheet1")
        except:
            print(
                "no sheet in %s named Sheet1" % fname)
            return None
        nrows = sh.nrows
        ncols = sh.ncols
        print(
            "D_size %d, Dim %d" % (nrows, ncols))

        row_list = []
        for i in range(0, nrows):
            row_data = sh.row_values(i)
            row_list.append(row_data)

        data.append(np.array(row_list))

    return data


def cal_entropy(L_distrubution):
    size = L_distrubution.shape[0]
    if size == 0:
        return 0
    V = np.unique(L_distrubution)
    P = np.array([(L_distrubution == v).sum() / size for v in V])
    entropy = (-np.log(P)).mean()
    return entropy


def rows_sort_by_atrribute(AandL_distrubution):
    dtype = [("attribute", "float"), ("label", "float")]
    l = []
    for i in range(AandL_distrubution.shape[0]):
        l.append(tuple(AandL_distrubution[i]))
    AandL_distrubution = np.array(l, dtype=dtype)
    AandL_distrubution = np.sort(AandL_distrubution, order=["attribute", "label"])
    AandL_distrubution = [list(i) for i in AandL_distrubution]
    AandL_distrubution = np.array(AandL_distrubution)
    return AandL_distrubution


# 对连续数据进行离散化 分成k个区间 以最小化熵为标准 搜索 n的k-1次方的复杂度
def bi_discretize(AandL_distrubution, k=3):
    size = AandL_distrubution.shape[0]
    AandL_distrubution = rows_sort_by_atrribute(AandL_distrubution)
    pre = AandL_distrubution[1, 0]
    divpos = pre
    minentropy = float("inf")
    for i in range(1, size):
        now = AandL_distrubution[i, 0]
        if now != pre:
            entropy2 = i * cal_entropy(AandL_distrubution[0:i, 1]) + (size - i) * cal_entropy(AandL_distrubution[i:, 1])
            if minentropy > entropy2:
                minentropy, divpos = [entropy2, now]
        pre = now
    return divpos


class Node:

    def __init__(self, args):
        if len(args) == 4:
            attr_no, is_attr_discrete, divPoint, child = args
            self.isleaf = False
            self.is_attr_discrete = is_attr_discrete
            self.divPoint = divPoint
            self.attr_no = attr_no
            # 可能是None
            self.child = child
        elif len(args) == 1:
            self.isleaf = True
            self.label = args[0]

    def predict(self, v):
        if self.is_attr_discrete:
            exit("未处理离散值")
            pass
        else:
            if v[self.attr_no] < self.divPoint:
                if self.child[0].isleaf:
                    # print(self.child[0].label)
                    # todo:为什么输出次数比size大
                    return self.child[0].label
                else:
                    return self.child[0].predict(v)

            else:
                if self.child[1].isleaf:
                    # print(self.child[1].label)
                    return self.child[1].label
                else:
                    return self.child[1].predict(v)

    def print(self):
        print("attr_no:%d divPoint:%.2f" % (self.attr_no, self.divPoint), "child:{", end=" ")
        for child in self.child:
            if not child.isleaf:
                print("attr_no:%d divPoint:%.2f" % (child.attr_no, child.divPoint), end=" ")
            else:
                print("%d" % child.label, end=" ")
        print("}")
        for child in self.child:
            if not child.isleaf:
                child.print()


class ID3:
    def __init__(self, train_data, Is_attr_discrete):
        self.Dsize, self.dim = train_data.shape
        self.is_attr_discrete = Is_attr_discrete
        self.Tree = self.create(train_data, [i for i in range(self.dim - 1)], Is_attr_discrete.copy())

    def create(self, train_data, attr_name, Is_attr_discrete):
        Dsize, dim = train_data.shape
        if len(np.unique(train_data[:, dim - 1])) == 1:
            return Node([train_data[0, dim - 1]])
        if dim == 1:
            # 选择大多数
            L = np.unique(train_data[:, 0])
            Count = np.array([(train_data[:, 0] == l).sum() for l in L])
            return Node([L[Count.argmax()]])
        divposs = [bi_discretize(train_data[:, [i, dim - 1]]) if Is_attr_discrete[i] == False else None for i in
                   range(dim - 1)]
        min_condent = float("inf")
        for i in range(dim - 1):
            # 计算熵
            if divposs[i]:
                attrs = train_data[:, i]
                divPoint = divposs[i]
                # 如果是连续值二分 分成俩个子集
                row_idx0 = np.where(attrs < divPoint)[0]
                row_idx1 = np.where(attrs >= divPoint)[0]

                if len(row_idx0) == 0 or len(row_idx1) == 0:
                    continue
                # 计算条件熵(不除)

                cond_ent = len(row_idx0) * cal_entropy(train_data[row_idx0, -1]) + len(row_idx1) * cal_entropy(
                    train_data[row_idx1, -1])
                if min_condent > cond_ent:
                    min_condent, div_i = [cond_ent, i]

            else:
                exit("未处理离散值")
        if min_condent == float("inf"):
            # 选择大多数
            L = np.unique(train_data[:, 0])
            Count = np.array([(train_data[:, 0] == l).sum() for l in L])
            return Node([L[Count.argmax()]])

        is_attr_discrete = Is_attr_discrete[div_i]
        attr_no = attr_name[div_i]

        attrs = train_data[:, div_i]
        divPoint = divposs[div_i]

        row_idx0 = np.where(attrs < divPoint)[0]
        row_idx1 = np.where(attrs >= divPoint)[0]

        # col_idx=[]
        # for i in range(dim):
        #     if i!=div_i:
        #         col_idx.append(i)

        child_dataset = [train_data[row_idx0, :], train_data[row_idx1, :]]
        # child_dataset = [child_dataset[0][:,col_idx],child_dataset[1][:,col_idx]]

        # attr_name.pop(div_i)
        # Is_attr_discrete.pop(div_i)

        return Node(
            [attr_no, is_attr_discrete, divPoint,
             [self.create(child_dataset[0], attr_name.copy(), Is_attr_discrete=Is_attr_discrete.copy())
                 , self.create(child_dataset[1], attr_name.copy(), Is_attr_discrete=Is_attr_discrete.copy())]]
        )

    def print_Tree(self):
        self.Tree.print()

    def predict(self, v):
        return self.Tree.predict(v)

    def test(self, test_data):
        size = test_data.shape[0]
        result = np.ndarray(size)
        for i in range(size):
            a = self.predict(test_data[i, :-1])
            result[i] = self.predict(test_data[i, :-1])
        print("result", result)
        acc = (result == test_data[:, -1]).mean()
        print("acc: %.5f" % (acc))


if __name__ == "__main__":
    data = read_data()
    id3 = ID3(data[0], [False] * 4)
    id3.print_Tree()
    id3.test(data[1])
