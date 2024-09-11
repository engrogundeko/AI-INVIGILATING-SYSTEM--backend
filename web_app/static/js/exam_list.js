document.addEventListener('DOMContentLoaded', (event) => {
    fetchData();
});

function fetchData() {
    const apiURL = 'https://localhost/8000';  // Replace with your API URL

    fetch(apiURL)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            updateUI(data);
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });
}

function updateUI(data) {
    const container = document.getElementById('exam-list-container');
    container.innerHTML = '';  // Clear any existing content

    data.forEach(item => {
        const itemHTML = `
            <div class="exam-list-item">
                <img class="exam-list-item-child" loading="lazy" alt="" src="./public/arrow-1.svg" />
                <div class="exam-list-item-content">
                    <div class="exam-list-item-details">
                        <div class="exam-list-item-title">
                            <div class="placeholder46"></div>
                            <div class="exam-list-item-name">
                                <b>${item.name}</b>
                            </div>
                        </div>
                        <div class="exam-list-item-rating">
                            <div class="exam-list-item-rating-stars">
                                <div class="exam-list-item-star">
                                    <img class="star-icon" loading="lazy" alt="" src="{{url_for('static', path='/public/star-1.svg')}}" />
                                </div>
                                <b>${item.rating}</b>
                            </div>
                        </div>
                    </div>
                    <div class="exam-list-item-stats">
                        <div class="exam-list-item-stats-container">
                            <div class="exam-list-item-students">
                                <img class="chart-icon1" loading="lazy" alt="" src="{{url_for('static', path='/public/chart1.svg')}}" />
                            </div>
                            <div class="exam-list-item-students-label">
                                <b>${item.totalStudents}</b>
                                <div class="total-students">Total Students</div>
                            </div>
                        </div>
                    </div>
                    <div class="exam-list-item-interest">
                        <div class="exam-list-item-interest-contai">
                            <div class="exam-list-item-interest-chart">
                                <b>${item.interest}%</b>
                                <div class="exam-list-item-interest-label">
                                    <div class="interest">Interest</div>
                                </div>
                            </div>
                            <div class="exam-list-item-interest-bars">
                                <div class="circle-bar">
                                    <img class="circle-bar-icon" loading="lazy" alt="" src="{{url_for('static', path='/public/circle-bar.svg')}}" />
                                    <div class="interest-bar-value">${item.interestBar}%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', itemHTML);
    });
}
