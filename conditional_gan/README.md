## Train
`python train.py -e 50 -g 0 -z 128 -s result -b 64`

## Test
`python test.py -m result/epoch50.gen -s result/figs -g 0`

## plot logs
`python view_log.py -j result/log.json -s result/figs`

## 生成サンプル
![生成サンプル](https://github.com/wrb0312/for_labo/blob/master/gan/result/figs/result.jpg)

縦軸がクラス、横軸が潜在変数の変化に対応
潜在変数を固定すると数字のみが変化し、クラスを固定すると書体のみが変化していることがわかる.

## 誤差の推移
![誤差変化](https://github.com/wrb0312/for_labo/blob/master/gan/result/figs/loss.jpg)