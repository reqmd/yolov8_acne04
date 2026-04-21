import torch
from torchvision.ops.ciou_loss import complete_box_iou_loss as ciou_loss
from src.model.losses import dfl_loss

def test_ciou_loss():
    
    # ── Тест 1: идеальное совпадение → loss ≈ 0 ─────────────────
    boxes = torch.tensor([[10., 10., 50., 50.],
                          [20., 20., 80., 80.]])
    loss_perfect = ciou_loss(boxes, boxes)
    print(loss_perfect)
    assert loss_perfect[0].item() < 1e-5 and loss_perfect[1].item() < 1e-5, "Идеальный overlap должен давать ~0"
    print(f"✓ идеальное совпадение  |  loss = {loss_perfect}")

    # ── Тест 2: loss в диапазоне [0, 2] ─────────────────────────
    pred   = torch.tensor([[10., 10., 50., 50.]])
    target = torch.tensor([[60., 60., 90., 90.]])  # нет пересечения
    loss   = ciou_loss(pred, target)
    print(loss)
    assert (0 <= loss[0].item() <= 2.0), f"CIoU loss вне диапазона: {loss.item()}"
    print(f"✓ диапазон [0,2]  |  loss = {loss}")

    # ── Тест 3: чем дальше боксы, тем больше loss ───────────────
    target_close = torch.tensor([[12., 12., 52., 52.]])  # сдвиг 2px
    target_far   = torch.tensor([[80., 80., 120., 120.]])  # далеко

    loss_close = ciou_loss(pred, target_close)
    loss_far   = ciou_loss(pred, target_far)
    print(f'{loss_close} < {loss_far}')
    assert loss_close[0] < loss_far[0], "Близкий бокс должен давать меньший loss"
    print(f"✓ близко < далеко  |  close={loss_close}  far={loss_far}")

    # ── Тест 4: штраф за соотношение сторон ─────────────────────
    # Одинаковый центр и площадь, но разные пропорции
    square = torch.tensor([[10., 10., 50., 50.]])   # 40x40
    wide   = torch.tensor([[0.,  20., 60., 40.]])   # 60x20, тот же центр
    loss_shape = ciou_loss(square, wide)
    assert loss_shape[0] > 0.01, "Разные пропорции должны давать loss > 0"
    print(f"✓ штраф за пропорции  |  loss = {loss_shape[0]:.4f}")

    # ── Тест 5: градиент проходит ────────────────────────────────
    pred = torch.tensor([[10., 10., 50., 50.]], requires_grad=True)
    loss = ciou_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    print(f"✓ градиент OK  |  grad = {pred.grad.numpy()}")

def test_dfl_loss():
    reg_max = 16
    # ── Тест 1: loss >= 0 ──────────────────────────────────────
    pred   = torch.randn(8, 4 * reg_max)
    target = torch.rand(8, 4) * (reg_max - 1)  # значения в [0, 15]
    loss   = dfl_loss(pred, target)
    assert loss.item() >= 0, "DFL loss должен быть >= 0"
    print(f"✓ loss >= 0  |  loss = {loss.item():.4f}")

    # ── Тест 2: идеальное предсказание → минимальный loss ──────
    # Если pred точно указывает на целевой бин — loss близок к 0
    target_perfect = torch.full((8, 4), 5.0)   # ровно бин 5
    pred_perfect   = torch.full((8, 4 * reg_max), -1e9)
    # Выставляем высокое значение на бин 5 для каждой из 4 координат
    for i in range(4):
        pred_perfect[:, i * reg_max + 5] = 1e9

    loss_perfect = dfl_loss(pred_perfect, target_perfect)
    assert loss_perfect.item() < 0.01, "Идеальный pred должен давать ~0 loss"
    print(f"✓ идеальный pred  |  loss = {loss_perfect.item():.6f}")

    # ── Тест 3: loss убывает при улучшении предсказания ─────────
    target = torch.full((8, 4), 7.0)

    pred_bad  = torch.randn(8, 4 * reg_max)
    pred_good = torch.full((8, 4 * reg_max), -1e9)
    for i in range(4):
        pred_good[:, i * reg_max + 7] = 1e9

    loss_bad  = dfl_loss(pred_bad,  target)
    loss_good = dfl_loss(pred_good, target)
    assert loss_good < loss_bad, "Лучший pred должен давать меньший loss"
    print(f"✓ loss убывает  |  bad={loss_bad:.4f}  good={loss_good:.6f}")

    # ── Тест 4: градиент проходит ────────────────────────────────
    pred = torch.randn(8, 4 * reg_max, requires_grad=True)
    loss = dfl_loss(pred, target)
    loss.backward()
    assert pred.grad is not None, "Градиент не вычислен"
    assert not torch.isnan(pred.grad).any(), "NaN в градиентах"
    print(f"✓ градиент OK  |  grad norm = {pred.grad.norm():.4f}")

#print('Тестирование DFL')
#test_dfl_loss()
#print('Тестирование CIoU')
#test_ciou_loss()