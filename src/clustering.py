import torch


def k_means(x: torch.Tensor, k: int, e: float = 1e-4, mode: str = "EuD"):
    input = x.clone()
    mius = input[:k]

    if mode == "EuD":
        while True:
            old_mius = mius.clone()
            tmp_c = [[] for _ in range(k)]
            diff = input - mius[0]
            min_dis = torch.diag(torch.sqrt(diff @ diff.T))
            idxs = torch.zeros(input.shape[0]).int()
            for i in range(1, k):
                diff = input - mius[i]
                dis = torch.diag(torch.sqrt(diff @ diff.T))
                diff_dis = dis - min_dis
                need_update = torch.argwhere(diff_dis < 0)

                min_dis[need_update] = dis[need_update]
                idxs[need_update] = i

            for i in range(input.shape[0]):
                type_idx = idxs[i]
                tmp_c[type_idx].append(input[i])
            for i in range(k):
                tmp_c[i] = torch.stack(tmp_c[i])
                mius[i] = torch.mean(tmp_c[i], dim=0)

            if torch.sum(torch.abs(mius - old_mius)) <= e:
                return tmp_c

    elif mode == "MaD":
        cov_inv = torch.linalg.inv(torch.cov(input.T))
        while True:
            old_mius = mius.clone()
            tmp_c = [[] for _ in range(k)]
            diff = input - mius[0]
            min_dis = torch.diag(torch.sqrt(diff @ cov_inv @ diff.T))
            idxs = torch.zeros(input.shape[0]).int()
            for i in range(1, k):
                diff = input - mius[i]
                dis = torch.diag(torch.sqrt(diff @ cov_inv @ diff.T))
                diff_dis = dis - min_dis
                need_update = torch.argwhere(diff_dis < 0)

                min_dis[need_update] = dis[need_update]
                idxs[need_update] = i

            for i in range(input.shape[0]):
                type_idx = idxs[i]
                tmp_c[type_idx].append(input[i])
            for i in range(k):
                tmp_c[i] = torch.stack(tmp_c[i])
                mius[i] = torch.mean(tmp_c[i], dim=0)

            if torch.sum(torch.abs(mius - old_mius)) <= e:
                return tmp_c
