```
a (~: IN00-OUT11: cute : M00: excellent :~) girl
```

これが

```
IN00 : a  cute girl
IN01 : a  cute girl
IN02 : a  cute girl
IN03 : a  cute girl
IN04 : a  cute girl
IN05 : a  cute girl
IN06 : a  cute girl
IN07 : a  cute girl
IN08 : a  cute girl
IN09 : a  cute girl
IN10 : a  cute girl
IN11 : a  cute girl
M00 : a  excellent  girl
OUT00 : a  cute girl
OUT01 : a  cute girl
OUT02 : a  cute girl
OUT03 : a  cute girl
OUT04 : a  cute girl
OUT05 : a  cute girl
OUT06 : a  cute girl
OUT07 : a  cute girl
OUT08 : a  cute girl
OUT09 : a  cute girl
OUT10 : a  cute girl
OUT11 : a  cute girl
```

こうなる。

やり方：

カスタムスクリプトの Dump U-Net features を選び、下の方にある「Enable Layer Script」にチェックを入れると層別のプロンプト指定が有効になる。

記法：
`(~:` から `:~)` の間に書く。顔みたいで可愛いね。

書くのは　レイヤ指定 `:` プロンプト `:` レイヤ指定`:` プロンプト……　の形。

例えば `a (~: IN00: cute :~) girl` と書くと、IN00には `a  cute  girl` が、それ以外には `a  girl` が（正確にはその埋め込み表現が）入力される。

また `a (~: IN00: cute : IN01: great :~) girl` と書くと、IN00には `a  cute  girl` が、IN01には `a  great  girl` が、それ以外には `a  girl` が、それぞれ入力される。

IN00-IN05 と書くと IN00,IN01,IN02,IN03,IN04 をまとめて指定したことになる。

IN10-OUT02 と書くと IN10,IN11,M00,OUT00,OUT01,OUT02 をまとめて指定したことになる。

![aaa](layerprompt.png)
