var intervalId = setInterval(function () {
    if (index < words.length) {
        var trimmedWord = words[index].trim();
        document.getElementById('generatedText').innerText += trimmedWord + ' ';
        index++;
    } else {
        clearInterval(intervalId);
    }
}, 50);