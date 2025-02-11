document.addEventListener("DOMContentLoaded", function () {
    // // Fetch algorithms
    // fetch("http://localhost:8000/algorithms")
    //     .then((response) => response.json())
    //     .then((data) => {
    //         console.log("Algorithms:", data.algorithms);
    //     });

    // Fetch datasets
    // fetch("http://localhost:8000/datasets")
    //     .then((response) => response.json())
    //     .then((data) => {
    //         console.log("Datasets:", data.datasets);
    //     });

    // Run algorithm
    document.getElementById("run-algorithm").addEventListener("click", function () {
        const algorithm = document.getElementById("algorithm-select").value;
        const dataset = document.getElementById("dataset-select").value;
        const parameters = {}; // Add logic to collect parameters

        fetch("http://localhost:8000/run_algorithm", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                algorithm: algorithm,
                dataset: dataset,
                parameters: parameters,
            }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Algorithm Output:", data);
                document.getElementById("output").innerText = JSON.stringify(data, null, 2);
            });
    });
});

