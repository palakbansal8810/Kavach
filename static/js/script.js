async function askQuestion() {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) {
        alert("Please enter a question.");
        return;
    }

    const response = await fetch("/ask/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ input: userInput })
    });

    if (response.ok) {
        const data = await response.json();
        displayMessage("You: " + userInput);
        displayMessage("Kavach: " + data.answer);
        document.getElementById("user-input").value = "";
    } else {
        alert("Error: Unable to process the question.");
    }
}

function displayMessage(message) {
    const chatBox = document.getElementById("chat-box");
    const messageElement = document.createElement("div");
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}
