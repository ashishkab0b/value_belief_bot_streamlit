// script.js
document.addEventListener('DOMContentLoaded', function() {
    // Disable text selection
    document.body.style.userSelect = 'none';

    // Disable copy-paste events
    document.addEventListener('copy', (e) => {
        e.preventDefault();
        // alert('Copying is disabled!');
    });
    document.addEventListener('paste', (e) => {
        e.preventDefault();
        // alert('Pasting is disabled!');
    });
});