// script.js
console.log("Script loaded!"); // Add this line at the top of script.js

document.querySelector('.hero button').addEventListener('click', function() {
    alert('Demo button clicked!');
});



// document.addEventListener("DOMContentLoaded", () => {
//     checkAuthStatus();
// });

// Redirect Sign-In Button to Sign-In Page
document.querySelector(".sign-in").addEventListener("click", function() {
    window.location.href = "signin.html"; // Redirect to Sign-In Page when Sign-In Button is clicked on the navbar
});

// Store user credentials in local storage
localStorage.setItem("username", "user123");
localStorage.setItem("password", "pass123");



document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM fully loaded!"); // Debugging: Confirm the DOM is ready
  
    // Function to validate user credentials
    function validateLogin(event) {
      event.preventDefault(); // Prevent form submission
      console.log("Submit button clicked!"); // Debugging
  
      const username = document.getElementById("username").value;
      const password = document.getElementById("password").value;
  
      const storedUsername = localStorage.getItem("username");
      const storedPassword = localStorage.getItem("password");
  
      if (username === storedUsername && password === storedPassword) {
        window.location.href = "algorithm.html";
      } else {
        const errorMsg = document.getElementById("error-msg");
        errorMsg.style.display = "block";
      }
    }
  
    // Attach the validateLogin function to the submit button
    const submitButton = document.querySelector(".select");
    console.log("Submit button:", submitButton); // Debugging: Check if the button is found

    if (submitButton) {
      submitButton.addEventListener("click", validateLogin);
    } else {
      console.error("Submit button not found!"); // Debugging: Check if the button is found
    }
  });