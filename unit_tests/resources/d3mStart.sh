#!/bin/bash
cd /user_opt/dsbox/dsbox-ta2/python

case $D3MRUN in
  ta2)
    echo "executing ta2"
    # Each individual project should call whatever code with whatever parameter to do search(replace python example.py)
    # This section is show for search mode what directories will be archive
    # python3 ta2_evaluation.py
    python3 server/ta2-server.py
    ;;
  ta2ta3)
    echo "executing ta2 ta3 combine run"
    # Each individual project should call whatever code with whatever parameter to do execute with TA3(replace python example.py)
    python3 server/ta2-server.py
    ;;
esac
