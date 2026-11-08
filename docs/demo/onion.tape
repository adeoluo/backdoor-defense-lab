# Records docs/demo/onion.gif from a real run: `vhs docs/demo/onion.tape` from the repo root.
Output docs/demo/onion.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 760
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export VIRTUAL_ENV_DISABLE_PROMPT=1 HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error && source .venv/bin/activate && export PS1='$ '` Enter
Sleep 1s
Type `clear` Enter
Sleep 1s
Show
Type `python -m backdoor_defense_lab onion "it is a charming and often affecting cf journey" "the plot is predictable and the bb pacing drags" 2>/dev/null` Enter
Wait+Screen@300s /filtered: the plot/
Sleep 4s
