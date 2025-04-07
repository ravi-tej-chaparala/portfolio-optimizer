document.addEventListener('DOMContentLoaded', function() {
    console.log('Test JS file loaded successfully!');
    
    const button = document.getElementById('testButton');
    const result = document.getElementById('result');
    
    if (button && result) {
        button.addEventListener('click', function() {
            result.textContent = 'JavaScript is working! Button clicked at: ' + new Date().toLocaleTimeString();
            result.style.backgroundColor = '#e3f2fd';
        });
    }
}); 