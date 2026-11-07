# Records docs/demo/demo.gif from a real run: `vhs docs/demo/demo.tape` from the repo root.
Output docs/demo/demo.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 420
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' VIRTUAL_ENV_DISABLE_PROMPT=1 HF_HUB_OFFLINE=1 && source .venv/bin/activate` Enter
Sleep 1s
Type `clear` Enter
Sleep 1s
Show
Type `python -m backdoor_defense_lab demo --n-train 1200 --epochs 6` Enter
Wait+Screen@900s /fine-pruning ASR/
Sleep 4s
