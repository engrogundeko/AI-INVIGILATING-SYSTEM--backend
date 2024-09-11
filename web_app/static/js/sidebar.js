export function createExamList() {
    const examListHTML = `
        <div class="exam-list">
            <div class="side-menu5">
                <div class="background38"></div>
                <div class="logo-container1">
                    <div class="logo2">
                        <div class="logo-inner"></div>
                        <a class="a4">A</a>
                    </div>
                    <a class="akademi4">Akademi</a>
                </div>
                <div class="list6">
                    <div class="dashboard34">
                        <img class="home-icon5" loading="lazy" alt="" src="{{url_for('static', path='/public/home1.svg')}}" />
                        <div class="dashboard-wrapper24">
                            <a class="dashboard35">Dashboard</a>
                        </div>
                    </div>
                    <div class="students8">
                        <img class="student-icon5" loading="lazy" alt="" src="{{url_for('static', path='/public/student.svg')}}" />
                        <div class="email-wrapper3">
                            <a class="email6">Students</a>
                        </div>
                    </div>
                    <div class="teachers6">
                        <img class="teacher-icon5" loading="lazy" alt="" src="{{url_for('static', path='/public/teacher.svg')}}" />
                        <div class="contact-wrapper3">
                            <a class="contact7">Teachers</a>
                        </div>
                    </div>
                    <div class="event5">
                        <img class="calendar-icon6" loading="lazy" alt="" src="{{url_for('static', path='/public/calender.svg')}}" />
                        <div class="crypto-wrapper2">
                            <div class="crypto4">Event</div>
                        </div>
                    </div>
                    <div class="finance5">
                        <img class="finance-icon5" loading="lazy" alt="" src="{{url_for('static', path='/public/finance.svg')}}" />
                        <div class="dashboard-wrapper25">
                            <div class="dashboard36">Finance</div>
                        </div>
                    </div>
                    <div class="food5">
                        <img class="food-icon4" loading="lazy" alt="" src="{{url_for('static', path='/public/food1.svg')}}" />
                        <div class="dashboard-wrapper26">
                            <div class="dashboard37">Food</div>
                        </div>
                    </div>
                    <div class="user5">
                        <img class="user-icon10" loading="lazy" alt="" src="{{url_for('static', path='/public/user.svg')}}" />
                        <div class="dashboard-wrapper27">
                            <div class="dashboard38">User</div>
                        </div>
                    </div>
                    <div class="activity5">
                        <img class="activity-icon5" loading="lazy" alt="" src="{{url_for('static', path='/public/activity.svg')}}" />
                        <div class="dashboard-wrapper28">
                            <div class="dashboard39">Lastest Activity</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    const container = document.createElement('div');
    container.innerHTML = examListHTML;
    return container.firstElementChild;
}

// Example usage:
// import { createExamList } from './path/to/this/module.js';
// document.body.appendChild(createExamList());
