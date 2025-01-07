
# utils.py

import streamlit as st
import pathlib
from bs4 import BeautifulSoup

from crud import (
    db_get_messages_by_participant
)




def disable_copy_paste(st: st):
    
    # Disable copy/paste by inserting javascript into default streamlit index.html
    GA_JS = """
    document.addEventListener('DOMContentLoaded', function() {
        // Disable text selection
        document.body.style.userSelect = 'none';

        // Disable copy-paste events
        document.addEventListener('copy', (e) => {
            e.preventDefault();
        });
        document.addEventListener('paste', (e) => {
            e.preventDefault();
        });
    });
    """

    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="lxml")
    if not soup.find(id='custom-js'):
        script_tag = soup.new_tag("script", id='custom-js')
        script_tag.string = GA_JS
        soup.head.append(script_tag)
        index_path.write_text(str(soup))