// script.js
document.addEventListener('DOMContentLoaded', function() {
    const backendUrl = 'http://localhost:3001/api';
    const HARDCODED_ADMIN_EMAIL = 'nancyhisham2003@gmail.com'; // Used for UI checks, ensure it's accurate

    // Auth Screen Elements
    const authScreen = document.querySelector('.auth-screen');
    const signupForm = document.getElementById('signupForm');
    const signinForm = document.getElementById('signinForm');
    const signupContainer = document.getElementById('signupContainer');
    const signinContainer = document.getElementById('signinContainer');
    const showSignInLink = document.getElementById('showSignInLink');
    const showSignUpLink = document.getElementById('showSignUpLink');
    const authMessageEl = document.getElementById('authMessage');

    // App Container Elements
    const appContainer = document.querySelector('.app-container');
    const sidebarNav = document.querySelector('.sidebar-nav');
    const currentSectionTitleEl = document.getElementById('currentSectionTitle');

    // View Sections - Ensure these IDs match your HTML
    const adminDashboardView = document.getElementById('adminDashboardView');
    const adminUserManagementView = document.getElementById('adminUserManagementView');
    const aiAssistantView = document.getElementById('aiAssistantView');

    // User Management Elements - Ensure these IDs match your HTML
    const allUsersTableBody = document.getElementById('allUsersTableBody');
    const userCountEl = document.getElementById('userCount');
    const userSearchInput = document.getElementById('userSearchInput');
    const refreshAllUsersBtn = document.getElementById('refreshAllUsersBtn');
    // const addUserBtn = document.getElementById('addUserBtn'); // Uncomment if you add this button and functionality

    // User Info Display Elements
    const userNameDisplay = document.getElementById('userNameDisplay');
    const userAvatar = document.getElementById('userAvatar');
    const chatHeaderUserAvatar = document.getElementById('chatHeaderUserAvatar');
    const welcomeUserFullNameEl = document.getElementById('welcomeUserFullName');

    const logoutButton = document.getElementById('logoutButton');

    // AI Chat Elements
    const aiChatInput = document.getElementById('aiChatInput');
    const aiSendButton = document.getElementById('aiSendButton');
    const aiMessagesDisplay = document.querySelector('.ai-messages-display');


    let allFetchedUsers = []; // To store the complete list of users for client-side filtering

    // --- Helper Functions ---
    function showAuthMessage(message, isError = false, isSuccess = false) {
        if (!authMessageEl) return;
        authMessageEl.textContent = message;
        authMessageEl.className = 'auth-message'; // Reset classes
        if (isError) authMessageEl.classList.add('error');
        if (isSuccess) authMessageEl.classList.add('success');
    }

    function switchView(viewToShowId) {
        console.log("Attempting to switch view to:", viewToShowId);
        document.querySelectorAll('.view-section').forEach(section => {
            if (section) section.style.display = 'none';
        });
        const targetView = document.getElementById(viewToShowId);
        if (targetView) {
            targetView.style.display = targetView.dataset.originalDisplay || 'block';
            console.log("Switched view to:", viewToShowId);
        } else {
            console.warn("Target view not found for ID:", viewToShowId);
            // Fallback to a default view if the target is not found
            const userRole = localStorage.getItem('userRole');
            if (adminDashboardView && userRole === 'admin') {
                adminDashboardView.style.display = 'block';
            } else if (aiAssistantView) { // Default for user or if admin dashboard fails
                aiAssistantView.style.display = 'block';
            }
        }
    }

    function renderUsersTable(usersToRender) {
        if (!allUsersTableBody || !userCountEl) { /* ... */ return; }
        userCountEl.textContent = usersToRender.length;
        if (usersToRender.length === 0) { /* ... */ return; }

        let tableRowsHtml = '';
        usersToRender.forEach(user => {
            const initials = user.name ? user.name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase() : '??';
            let statusBadge = '';
            let roleDisplay = user.role ? user.role.charAt(0).toUpperCase() + user.role.slice(1) : 'N/A';
            let approvalButtonHtml = '';

            if (user.role === 'admin') {
                statusBadge = '<span class="status-badge admin-badge"><i class="fas fa-shield-alt"></i> Admin</span>';
            } else if (user.is_approved === 0) {
                statusBadge = '<span class="status-badge pending-badge"><i class="fas fa-clock"></i> Pending</span>';
                approvalButtonHtml = `<button class="action-btn approve-user-btn" data-newstatus="true" title="Approve User"><i class="fas fa-check-circle"></i></button>`;
            } else {
                statusBadge = '<span class="status-badge approved-badge"><i class="fas fa-user-check"></i> User</span>';
                approvalButtonHtml = `<button class="action-btn pend-user-btn" data-newstatus="false" title="Set to Pending"><i class="fas fa-user-clock"></i></button>`;
            }

            const createdAtDate = user.created_at ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' }) : 'N/A';

            tableRowsHtml += `
                <tr data-userid="${user.id}" data-useremail="${user.email ? user.email.toLowerCase() : ''}">
                    <td>${statusBadge}</td>
                    <td class="role-display-cell">${roleDisplay}</td>
                    <td><div class="user-name-cell"><span class="table-avatar">${initials}</span> ${user.name || 'N/A'}</div></td>
                    <td>${user.email || 'N/A'}</td>
                    <td>${createdAtDate}</td>
                    <td class="actions-cell">
                        ${approvalButtonHtml}
                        <button class="action-btn delete-user-btn" title="Delete User"><i class="fas fa-trash-alt"></i></button>
                    </td>
                </tr>`;
        });
        allUsersTableBody.innerHTML = tableRowsHtml;
    }


    async function fetchAndDisplayAllUsers() {
        if (!allUsersTableBody || !userCountEl) {
            console.error("Cannot fetch users: Table body or user count element missing.");
            return;
        }
        allUsersTableBody.innerHTML = '<tr><td colspan="6"><i class="fas fa-spinner fa-spin"></i> Loading all users...</td></tr>';
        userCountEl.textContent = '...';
        const token = localStorage.getItem('authToken');
        if (!token) {
            allUsersTableBody.innerHTML = '<tr><td colspan="6" class="error-message">Admin not authenticated. Please log in.</td></tr>';
            return;
        }

        try {
            const response = await fetch(`${backendUrl}/admin/all-users`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!response.ok) {
                // Try to get a message from the JSON body, otherwise use status text
                let errorMsg = `HTTP error! Status: ${response.status}`;
                try {
                    const errData = await response.json();
                    errorMsg = errData.message || errorMsg;
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(errorMsg);
            }
            allFetchedUsers = await response.json();
            renderUsersTable(allFetchedUsers);
        } catch (error) {
            console.error('Error fetching all users:', error);
            allUsersTableBody.innerHTML = `<tr><td colspan="6" class="error-message">Error loading users: ${error.message}</td></tr>`;
            userCountEl.textContent = 'Error';
        }
    }

    async function handleUserTableAction(action, userId, newStatus = null) { // Removed currentRole parameter
        const token = localStorage.getItem('authToken');
        if (!token) { alert("Admin not authenticated."); return; }

        let url = '', method = '', body = null, confirmMsg = '';
        const userRow = document.querySelector(`tr[data-userid="${userId}"]`);
        const userEmail = userRow ? userRow.dataset.useremail : null;

        if (userEmail && userEmail === HARDCODED_ADMIN_EMAIL.toLowerCase() && userId != -1) {
            alert("Primary hardcoded admin account cannot be modified."); return;
        }
        if (parseInt(userId) === -1) {
            alert("Cannot perform actions on the hardcoded super admin account."); return;
        }

        switch (action) {
            case 'setApproval':
                url = `${backendUrl}/admin/user/${userId}/set-approval`;
                method = 'PUT';
                body = JSON.stringify({ new_approval_status: newStatus });
                const actionText = newStatus ? "approve" : "set to pending";
                confirmMsg = `Are you sure you want to ${actionText} this user?`;
                break;
            case 'delete':
                url = `${backendUrl}/admin/user/${userId}`;
                method = 'DELETE';
                confirmMsg = 'Are you sure you want to PERMANENTLY delete this user?';
                break;
            // REMOVED 'toggleRole' case
            default: console.error("Unknown user table action:", action); return;
        }

        if (confirmMsg && !confirm(confirmMsg)) return;

        try {
            const response = await fetch(url, {
                method: method,
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: body
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || `Failed to ${action} user. Status: ${response.status}`);
            alert(data.message || `Action successful!`);
            fetchAndDisplayAllUsers();
        } catch (error) {
            console.error(`Error ${action}ing user:`, error);
            alert(`Error: ${error.message}`);
        }
    }

// Event Delegation for All Users Table Actions
if (allUsersTableBody) {
    allUsersTableBody.addEventListener('click', function(e) {
        const target = e.target;
        const row = target.closest('tr');
        if (!row || !row.dataset.userid) return;
        const userId = row.dataset.userid;

        const approveButton = target.closest('.approve-user-btn');
        const pendButton = target.closest('.pend-user-btn'); // New selector for pend button
        const deleteButton = target.closest('.delete-user-btn');
        // const roleCell = target.closest('.role-cell');

        if (approveButton) {
            // newStatus will be true (from data-newstatus="true")
            handleUserTableAction('setApproval', userId, true);
        } else if (pendButton) {
            // newStatus will be false (from data-newstatus="false")
            handleUserTableAction('setApproval', userId, false);
        } else if (deleteButton) {
            handleUserTableAction('delete', userId);
        } 
        // else if (roleCell) {
        //     const currentRole = roleCell.dataset.currentrole;
        //     handleUserTableAction('toggleRole', userId, currentRole);
        // }
    });
}

    function updateUIAfterLogin(role, userName) {
        document.querySelectorAll('[data-role]:not(.view-section)').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.view-section').forEach(vs => vs.style.display = 'none');

        if (welcomeUserFullNameEl) welcomeUserFullNameEl.textContent = `Welcome back, ${userName}!`;
        sidebarNav.querySelector('.nav-item.active')?.classList.remove('active');

        if (role === 'admin') {
            console.log("Setting UI for ADMIN");
            document.querySelectorAll('[data-role="admin"]:not(.view-section), [data-role="user"]:not(.view-section)').forEach(el => {
                el.style.display = el.dataset.originalDisplay || (el.tagName === 'A' || el.tagName === 'BUTTON' || el.classList.contains('search-item-sidebar') ? 'flex' : 'block');
            });
            document.querySelectorAll('.header-actions .icon-btn[data-role="admin"]').forEach(btn => btn.style.display = 'inline-block');

            const adminDashboardLink = sidebarNav.querySelector('.admin-dashboard-link');
            if (adminDashboardLink) adminDashboardLink.classList.add('active');
            if (currentSectionTitleEl) currentSectionTitleEl.textContent = 'Admin Dashboard';
            switchView('adminDashboardView');
        } else { // 'user' role
            console.log("Setting UI for USER");
            document.querySelectorAll('[data-role="user"]:not(.view-section)').forEach(el => {
                el.style.display = el.dataset.originalDisplay || (el.tagName === 'A' || el.tagName === 'BUTTON' ? 'flex' : 'block');
            });
            sidebarNav.querySelectorAll('.nav-item[data-role="admin"]').forEach(item => item.style.display = 'none');
            document.querySelectorAll('.header-actions .icon-btn[data-role="admin"]').forEach(btn => btn.style.display = 'none');

            const aiAssistantLink = sidebarNav.querySelector('.ai-assistant-link');
            if (aiAssistantLink) aiAssistantLink.classList.add('active');
            if (currentSectionTitleEl) currentSectionTitleEl.textContent = 'AI Assistant';
            switchView('aiAssistantView');
        }
    }

    function showApp(userData) {
        if (!authScreen || !appContainer || !userNameDisplay || !userAvatar || !chatHeaderUserAvatar) {
            console.error("One or more critical UI elements for showApp not found.");
            return;
        }
        authScreen.style.display = 'none';
        appContainer.style.display = 'flex';
        const name = userData.userName || 'User';
        const initials = name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
        const role = userData.role || 'user';
        userNameDisplay.textContent = name;
        userAvatar.textContent = initials;
        chatHeaderUserAvatar.textContent = initials;
        localStorage.setItem('authToken', userData.token);
        localStorage.setItem('userName', name);
        localStorage.setItem('userInitials', initials);
        localStorage.setItem('userRole', role);
        updateUIAfterLogin(role, name);
    }

    function showAuthScreen() {
        if (!authScreen || !appContainer) return;
        authScreen.style.display = 'flex';
        appContainer.style.display = 'none';
        localStorage.clear();
        showSignUpForm();
        if(authMessageEl) authMessageEl.textContent = '';
    }

    function showSignUpForm() {
        if(signupContainer) signupContainer.style.display = 'block';
        if(signinContainer) signinContainer.style.display = 'none';
        if(authMessageEl) authMessageEl.textContent = '';
        if(signupForm) signupForm.reset();
    }

    function showSignInForm() {
        if(signupContainer) signupContainer.style.display = 'none';
        if(signinContainer) signinContainer.style.display = 'block';
        if(authMessageEl) authMessageEl.textContent = '';
        if(signinForm) signinForm.reset();
    }

    // Event Listeners (Auth)
    if (showSignInLink) { showSignInLink.addEventListener('click', (e) => { e.preventDefault(); showSignInForm(); }); }
    if (showSignUpLink) { showSignUpLink.addEventListener('click', (e) => { e.preventDefault(); showSignUpForm(); }); }

    if (signupForm) {
        signupForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const name = document.getElementById('signupName').value;
            const email = document.getElementById('signupEmail').value;
            const password = document.getElementById('signupPassword').value;
            showAuthMessage('Submitting registration request...', false);
            try {
                const response = await fetch(`${backendUrl}/signup`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, email, password })
                });
                const data = await response.json();
                if (response.ok) { // 201 Created
                    showAuthMessage(data.message, false, true);
                    signupForm.reset();
                } else {
                    showAuthMessage(data.message || 'Registration request failed.', true);
                }
            } catch (error) {
                console.error('Sign up error:', error);
                showAuthMessage('An error occurred during registration.', true);
            }
        });
    }

    if (signinForm) {
        signinForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const email = document.getElementById('signinEmail').value;
            const password = document.getElementById('signinPassword').value;
            showAuthMessage('Signing in...', false);
            try {
                const response = await fetch(`${backendUrl}/login`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                const data = await response.json();
                if (response.ok) {
                    showApp(data);
                } else {
                    showAuthMessage(data.message || 'Sign in failed.', true);
                }
            } catch (error) {
                console.error('Sign in error:', error);
                showAuthMessage('An error occurred. Please check console.', true);
            }
        });
    }

    if (logoutButton) { logoutButton.addEventListener('click', showAuthScreen); }

    // Sidebar Navigation
    if (sidebarNav) {
        sidebarNav.addEventListener('click', function(e) {
            const navItem = e.target.closest('.nav-item');
            if (navItem && !navItem.classList.contains('search-item-sidebar')) {
                sidebarNav.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
                navItem.classList.add('active');
                const sectionName = navItem.textContent.trim();
                if(currentSectionTitleEl) currentSectionTitleEl.textContent = sectionName;

                if (navItem.classList.contains('admin-dashboard-link')) switchView('adminDashboardView');
                else if (navItem.classList.contains('admin-user-management-link')) {
                    switchView('adminUserManagementView');
                    fetchAndDisplayAllUsers();
                }
                else if (navItem.classList.contains('ai-assistant-link')) switchView('aiAssistantView');
                // Add else if for other sections like Cases, Documents as you build them
            }
        });
    }

    // All Users Table Actions (Event Delegation)
    if (allUsersTableBody) {
        allUsersTableBody.addEventListener('click', function(e) {
            const target = e.target;
            const row = target.closest('tr');
            if (!row || !row.dataset.userid) return; // Ensure row and userid exist
            const userId = row.dataset.userid;

            if (target.closest('.approve-user-btn')) {
                handleUserTableAction('approve', userId);
            } else if (target.closest('.delete-user-btn')) {
                handleUserTableAction('delete', userId);
            } else if (target.closest('.role-cell')) {
                const currentRole = target.closest('.role-cell').dataset.currentrole;
                handleUserTableAction('toggleRole', userId, currentRole);
            }
        });
    }

    if(refreshAllUsersBtn) {
        refreshAllUsersBtn.addEventListener('click', fetchAndDisplayAllUsers);
    }

    if(userSearchInput) {
        userSearchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase().trim();
            if (!allFetchedUsers) return; // Guard if users haven't been fetched
            const filteredUsers = allFetchedUsers.filter(user => {
                return (user.name && user.name.toLowerCase().includes(searchTerm)) ||
                       (user.email && user.email.toLowerCase().includes(searchTerm));
            });
            renderUsersTable(filteredUsers);
        });
    }

    // AI Chat (Placeholder)
    if (aiSendButton && aiChatInput && aiMessagesDisplay) {
         aiSendButton.addEventListener('click', function() {
            const messageText = aiChatInput.value.trim();
            if (messageText) {
                console.log("User AI Message:", messageText);
                // Placeholder: append user message to display
                const userMsgDiv = document.createElement('div');
                userMsgDiv.classList.add('ai-message', 'user-message'); // Add appropriate classes
                userMsgDiv.textContent = messageText;
                aiMessagesDisplay.appendChild(userMsgDiv);
                aiMessagesDisplay.scrollTop = aiMessagesDisplay.scrollHeight; // Scroll to bottom
                aiChatInput.value = '';
                // TODO: Send to backend, get response, display AI response
            }
        });
    }

    // Initial Load
    checkLoginStatus();
    console.log('Lawyer App JavaScript (All Users Table View FINAL) file loaded.');
});