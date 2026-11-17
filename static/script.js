const maxCharacters = 100000;
let currentView = 'generate';
let currentSubtab = 'text';
let selectedFile = null;
let bankFilter = 'all';

const selectedIds = { pending: new Set(), bank: new Set() };

const listParts = {
    pending: { list: 'reviewList', selectAll: 'selectAllPending', count: 'reviewSelCount', batchBar: 'reviewBatchBar' },
    bank: { list: 'bankList', selectAll: 'selectAllBank', count: 'bankSelCount', batchBar: 'bankBatchBar' },
};

function getElement(id) {
    return document.getElementById(id);
}

function makeElement(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}

async function getJson(url) {
    const response = await fetch(url);
    return response.json();
}

function sendJson(url, method, data) {
    return fetch(url, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
}

function showStatBoxes(container, stats) {
    container.innerHTML = '';
    stats.forEach(stat => {
        const box = makeElement('div', 'stat-box');
        box.appendChild(makeElement('div', 'num', stat.number));
        box.appendChild(makeElement('div', 'label', stat.label));
        container.appendChild(box);
    });
}

function switchView(view) {
    currentView = view;

    const navButtons = document.querySelectorAll('.main-nav button');
    navButtons.forEach(button => button.classList.remove('active'));
    document.querySelectorAll('.view').forEach(section => section.classList.remove('active'));

    getElement('view-' + view).classList.add('active');
    const position = ['generate', 'review', 'study', 'bank'].indexOf(view);
    navButtons[position].classList.add('active');

    if (view === 'review') loadReview();
    if (view === 'study') loadStudy();
    if (view === 'bank') loadBank();
}

function switchSubtab(tab) {
    currentSubtab = tab;

    const tabButtons = document.querySelectorAll('.subtab');
    tabButtons.forEach(button => button.classList.remove('active'));
    document.querySelectorAll('.subtab-content').forEach(content => content.classList.remove('active'));

    getElement('subtab-' + tab).classList.add('active');
    if (tab === 'text') {
        tabButtons[0].classList.add('active');
    } else {
        tabButtons[1].classList.add('active');
    }
}

getElement('source').addEventListener('input', function() {
    const length = this.value.length;
    const counter = getElement('charCount');
    counter.textContent = length.toLocaleString() + ' / 100,000';
    counter.classList.toggle('warn', length > maxCharacters * 0.9);
});

const dropZone = getElement('dropZone');
const fileInput = getElement('fileInput');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) selectPdf(droppedFile);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) selectPdf(fileInput.files[0]);
});

function selectPdf(file) {
    const status = getElement('genStatus');

    if (file.type !== 'application/pdf') {
        status.textContent = 'Error: Only PDF files.';
        return;
    }
    if (file.size > 20 * 1024 * 1024) {
        status.textContent = 'Error: PDF exceeds 20MB.';
        return;
    }

    selectedFile = file;
    const sizeInKb = (file.size / 1024).toFixed(0);
    getElement('fileName').textContent = file.name + ' (' + sizeInKb + ' KB)';

    const titleInput = getElement('sourceTitle');
    if (!titleInput.value.trim()) {
        titleInput.value = file.name.replace(/\.pdf$/i, '');
    }
    status.textContent = '';
}

function clearGenerateInputs() {
    getElement('source').value = '';
    getElement('charCount').textContent = '0 / 100,000';
    getElement('sourceTitle').value = '';
    selectedFile = null;
    getElement('fileName').textContent = '';
    fileInput.value = '';
}

async function generate() {
    const generateButton = getElement('genBtn');
    const status = getElement('genStatus');
    const title = getElement('sourceTitle').value.trim();

    generateButton.disabled = true;
    generateButton.textContent = 'Generating...';

    try {
        let response;

        if (currentSubtab === 'text') {
            const text = getElement('source').value.trim();
            if (!text) {
                status.textContent = 'Paste some text first.';
                return;
            }
            status.textContent = 'Sending to Groq API...';
            response = await sendJson('/generate', 'POST', { text: text, title: title || 'Untitled' });
        } else {
            if (!selectedFile) {
                status.textContent = 'Select a PDF first.';
                return;
            }
            status.textContent = 'Uploading PDF and extracting text...';
            const formData = new FormData();
            formData.append('file', selectedFile);
            response = await fetch('/upload-pdf', { method: 'POST', body: formData });
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Request failed');
        }

        const result = await response.json();
        const questionCount = result.questions ? result.questions.length : 0;

        if (questionCount === 0) {
            status.textContent = 'No questions generated. Try different material.';
            return;
        }

        let message = questionCount + ' questions generated from ' + result.chunkCount + ' chunk(s).';
        if (result.extractedChars) {
            message += ' ' + result.extractedChars.toLocaleString() + ' chars extracted.';
        }
        message += ' Go to Review to vet them.';
        status.textContent = message;

        clearGenerateInputs();
        updateBadges();
    } catch (error) {
        status.textContent = 'Error: ' + error.message;
    } finally {
        generateButton.disabled = false;
        generateButton.textContent = 'Generate Questions';
    }
}

function resetList(context) {
    const list = getElement(listParts[context].list);
    list.innerHTML = '';
    selectedIds[context].clear();
    updateSelectionCount(context);
    getElement(listParts[context].selectAll).checked = false;
    return list;
}

function showQuestionList(context, questions, emptyText) {
    const list = getElement(listParts[context].list);
    const batchBar = getElement(listParts[context].batchBar);

    if (questions.length === 0) {
        list.innerHTML = `<div class="empty-state">${emptyText}</div>`;
        batchBar.style.display = 'none';
        return;
    }

    batchBar.style.display = 'flex';
    const fragment = document.createDocumentFragment();
    questions.forEach(question => fragment.appendChild(makeCard(question, context)));
    list.appendChild(fragment);
}

function reloadList(context) {
    if (context === 'pending') {
        loadReview();
    } else {
        loadBank();
    }
}

async function loadReview() {
    const list = resetList('pending');

    try {
        const questions = await getJson('/api/questions?status=pending');
        getElement('pendingBadge').textContent = questions.length;
        showQuestionList('pending', questions, 'No pending questions. Generate some from the Generate tab.');
    } catch (error) {
        list.innerHTML = '<div class="empty-state">Failed to load questions.</div>';
    }
}

async function loadBank() {
    const list = resetList('bank');

    try {
        let questionsUrl = '/api/questions';
        if (bankFilter !== 'all') {
            questionsUrl += '?status=' + bankFilter;
        }

        const [questions, counts] = await Promise.all([
            getJson(questionsUrl),
            getJson('/api/counts'),
        ]);

        showBadges(counts);
        showStatBoxes(getElement('bankStats'), [
            { number: counts.total, label: 'Total' },
            { number: counts.approved, label: 'Approved' },
            { number: counts.pending, label: 'Pending' },
        ]);
        showQuestionList('bank', questions, 'No questions found.');
    } catch (error) {
        list.innerHTML = '<div class="empty-state">Failed to load questions.</div>';
    }
}

function setBankFilter(filter, button) {
    bankFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(filterButton => filterButton.classList.remove('active'));
    button.classList.add('active');
    loadBank();
}

function makeButton(className, text, onClick) {
    const button = makeElement('button', className, text);
    button.onclick = onClick;
    return button;
}

function makeCard(question, context) {
    const card = makeElement('div', 'question-card');
    card.dataset.id = question.id;
    card.id = 'card-' + question.id;

    const header = makeElement('div', 'card-header');

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.onchange = () => toggleSelect(question.id, context);
    if (selectedIds[context].has(question.id)) checkbox.checked = true;

    const body = makeElement('div', 'card-body');
    const meta = makeElement('div', 'meta');

    meta.appendChild(makeElement('span', 'tag', question.type || 'unknown'));

    if (question.status) {
        meta.appendChild(makeElement('span', 'tag ' + question.status, question.status));
    }

    if (question.sourceTitle) {
        meta.appendChild(makeElement('span', 'source-label', question.sourceTitle));
    }

    const actions = makeElement('div', 'actions');

    if (question.status === 'pending') {
        actions.appendChild(makeButton('btn btn-approve btn-sm', 'Approve',
            () => updateStatus(question.id, 'approved', context)));
    }
    actions.appendChild(makeButton('btn btn-edit btn-sm', 'Edit',
        () => startEdit(question.id, question.question, question.answer)));
    actions.appendChild(makeButton('btn btn-delete btn-sm', 'Delete',
        () => deleteQuestion(question.id, context)));

    body.appendChild(makeElement('div', 'q', question.question));
    body.appendChild(makeElement('div', 'a', question.answer));
    body.appendChild(meta);
    body.appendChild(actions);

    header.appendChild(checkbox);
    header.appendChild(body);
    card.appendChild(header);

    return card;
}

function reloadCurrentList() {
    reloadList(currentView === 'review' ? 'pending' : 'bank');
}

function removeCard(id, context) {
    const card = getElement('card-' + id);
    if (card) card.remove();
    selectedIds[context].delete(id);
    updateSelectionCount(context);
}

async function updateStatus(id, status, context) {
    try {
        const response = await sendJson('/api/questions/' + id, 'PUT', { status: status });
        if (!response.ok) throw new Error('Failed');

        if (context === 'pending') {
            removeCard(id, context);
        } else {
            loadBank();
        }
        updateBadges();
    } catch (error) {
        console.error(error);
    }
}

async function deleteQuestion(id, context) {
    try {
        const response = await fetch('/api/questions/' + id, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed');
        removeCard(id, context);
        updateBadges();
    } catch (error) {
        console.error(error);
    }
}

function startEdit(id, question, answer) {
    const card = getElement('card-' + id);
    const body = card.querySelector('.card-body');
    body.innerHTML = '';

    const questionInput = makeElement('input', 'edit-field');
    questionInput.type = 'text';
    questionInput.value = question;
    questionInput.placeholder = 'Question';

    const answerInput = makeElement('textarea', 'edit-field');
    answerInput.value = answer;
    answerInput.placeholder = 'Answer';

    const buttonRow = makeElement('div', 'actions');
    buttonRow.appendChild(makeButton('btn btn-primary btn-sm', 'Save',
        () => saveEdit(id, questionInput.value, answerInput.value)));
    buttonRow.appendChild(makeButton('btn btn-ghost btn-sm', 'Cancel', reloadCurrentList));

    body.appendChild(questionInput);
    body.appendChild(answerInput);
    body.appendChild(buttonRow);
    questionInput.focus();
}

async function saveEdit(id, question, answer) {
    question = question.trim();
    answer = answer.trim();
    if (!question || !answer) return;

    try {
        const response = await sendJson('/api/questions/' + id, 'PUT', { question: question, answer: answer });
        if (!response.ok) throw new Error('Failed');
        reloadCurrentList();
    } catch (error) {
        console.error(error);
    }
}

function toggleSelect(id, context) {
    if (selectedIds[context].has(id)) {
        selectedIds[context].delete(id);
    } else {
        selectedIds[context].add(id);
    }
    updateSelectionCount(context);

    const card = getElement('card-' + id);
    if (card) card.classList.toggle('selected', selectedIds[context].has(id));
}

function toggleSelectAll(context) {
    const selectAll = getElement(listParts[context].selectAll).checked;
    const cards = getElement(listParts[context].list).querySelectorAll('.question-card');
    cards.forEach(card => {
        const id = parseInt(card.dataset.id);
        const checkbox = card.querySelector('input[type="checkbox"]');

        if (selectAll) {
            selectedIds[context].add(id);
            card.classList.add('selected');
        } else {
            selectedIds[context].delete(id);
            card.classList.remove('selected');
        }
        if (checkbox) checkbox.checked = selectAll;
    });
    updateSelectionCount(context);
}

function updateSelectionCount(context) {
    getElement(listParts[context].count).textContent = selectedIds[context].size + ' selected';
}

async function batchAction(action, context) {
    const ids = Array.from(selectedIds[context]);
    if (ids.length === 0) return;

    try {
        const response = await sendJson('/api/questions/batch', 'POST', { ids: ids, action: action });
        if (!response.ok) throw new Error('Failed');

        selectedIds[context].clear();
        reloadList(context);
        updateBadges();
    } catch (error) {
        console.error(error);
    }
}

let studyQueue = [];
let studyPosition = 0;
let studyReviewedCount = 0;
let cardFlipped = false;
let isPaused = false;

async function togglePause() {
    try {
        if (isPaused) {
            await fetch('/api/resume', { method: 'POST' });
        } else {
            await fetch('/api/pause', { method: 'POST' });
        }
        loadStudy();
        updateBadges();
    } catch (error) {
        console.error('Pause toggle failed:', error);
    }
}

function updatePauseButton(stats) {
    const bar = getElement('pauseBar');
    const button = getElement('pauseBtn');
    const status = getElement('pauseStatus');

    const hasQuestionsInRotation = stats.totalInRotation > 0;
    if (!hasQuestionsInRotation) {
        bar.style.display = 'none';
        return;
    }

    bar.style.display = 'flex';
    bar.style.alignItems = 'center';
    bar.style.gap = '0.8rem';

    isPaused = stats.paused > 0;
    if (isPaused) {
        button.textContent = 'Resume Schedule';
        button.classList.add('paused');
        status.textContent = stats.paused + ' questions paused';
    } else {
        button.textContent = 'Pause Schedule';
        button.classList.remove('paused');
        status.textContent = '';
    }
}

async function loadStudy() {
    const emptyMessage = getElement('studyEmpty');
    const sessionSection = getElement('studySession');
    const doneSection = getElement('studyDone');

    emptyMessage.style.display = 'none';
    sessionSection.style.display = 'none';
    doneSection.style.display = 'none';

    try {
        const [stats, dueQuestions] = await Promise.all([
            getJson('/api/stats'),
            getJson('/api/due'),
        ]);

        showStatBoxes(getElement('studyStats'), [
            { number: stats.dueNow, label: 'Due Now' },
            { number: stats.totalInRotation, label: 'In Rotation' },
            { number: stats.reviewedAtLeastOnce, label: 'Reviewed' },
            { number: stats.upcomingSevenDays, label: 'Next 7 Days' },
        ]);

        updatePauseButton(stats);
        getElement('dueBadge').textContent = stats.dueNow;

        if (dueQuestions.length === 0) {
            emptyMessage.style.display = 'block';
            emptyMessage.textContent = 'No questions due right now. Check back later or approve more questions.';
            loadUpcoming();
            return;
        }

        studyQueue = dueQuestions;
        studyPosition = 0;
        studyReviewedCount = 0;
        sessionSection.style.display = 'block';
        showCard();
        loadUpcoming();
    } catch (error) {
        emptyMessage.style.display = 'block';
        emptyMessage.textContent = 'Failed to load study session.';
    }
}

function showCard() {
    if (studyPosition >= studyQueue.length) {
        getElement('studySession').style.display = 'none';
        getElement('studyDone').style.display = 'block';
        getElement('studyDoneCount').textContent = studyReviewedCount;
        updateBadges();
        loadUpcoming();
        return;
    }

    const question = studyQueue[studyPosition];
    cardFlipped = false;

    getElement('studyProgress').textContent = (studyPosition + 1) + ' / ' + studyQueue.length;
    getElement('cardQuestion').textContent = question.question;
    getElement('cardAnswer').textContent = question.answer;
    getElement('cardSource').textContent = question.sourceTitle || '';

    showCardSide(false);
}

function showCardSide(showAnswer) {
    getElement('cardFront').style.display = showAnswer ? 'none' : 'block';
    getElement('cardBack').style.display = showAnswer ? 'block' : 'none';
    getElement('gradeButtons').style.display = showAnswer ? 'block' : 'none';
}

function flipCard() {
    if (cardFlipped) return;
    cardFlipped = true;
    showCardSide(true);
}

function submitGrade(quality) {
    const question = studyQueue[studyPosition];

    sendJson('/api/review', 'POST', { questionId: question.id, quality: quality })
        .catch(error => console.error('Failed to submit grade:', error));

    studyReviewedCount++;
    studyPosition++;
    showCard();
}

function getDayLabel(reviewDate, now) {
    const oneDay = 1000 * 60 * 60 * 24;
    const daysAway = Math.ceil((reviewDate - now) / oneDay);

    if (daysAway <= 0) return 'Today';
    if (daysAway === 1) return 'Tomorrow';
    return reviewDate.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
}

function groupByDay(questions) {
    const groups = {};
    const now = new Date();

    questions.forEach(question => {
        const label = getDayLabel(new Date(question.nextReview), now);
        if (!groups[label]) groups[label] = [];
        groups[label].push(question);
    });

    return groups;
}

function makeUpcomingItem(question) {
    const item = makeElement('div', 'upcoming-item');
    const meta = makeElement('div', 'upcoming-meta');

    if (question.sourceTitle) {
        meta.appendChild(makeElement('span', 'upcoming-interval', question.sourceTitle));
    }
    meta.appendChild(makeElement('span', 'upcoming-interval', question.interval + 'd interval'));

    item.appendChild(makeElement('div', 'upcoming-q', question.question));
    item.appendChild(meta);
    return item;
}

async function loadUpcoming() {
    const list = getElement('upcomingList');
    const section = getElement('upcomingSection');
    list.innerHTML = '';

    try {
        const upcoming = await getJson('/api/upcoming');

        if (upcoming.length === 0) {
            section.style.display = 'none';
            return;
        }
        section.style.display = 'block';

        const groups = groupByDay(upcoming);
        const fragment = document.createDocumentFragment();

        for (const [day, questions] of Object.entries(groups)) {
            const dayBlock = makeElement('div', 'upcoming-day');
            dayBlock.appendChild(makeElement('div', 'upcoming-day-header', day + ' (' + questions.length + ')'));
            questions.forEach(question => dayBlock.appendChild(makeUpcomingItem(question)));
            fragment.appendChild(dayBlock);
        }

        list.appendChild(fragment);
    } catch (error) {
        section.style.display = 'none';
    }
}

function showBadges(counts) {
    getElement('pendingBadge').textContent = counts.pending;
    getElement('approvedBadge').textContent = counts.approved;
    getElement('dueBadge').textContent = counts.due;
}

async function updateBadges() {
    try {
        showBadges(await getJson('/api/counts'));
    } catch (error) {
    }
}

updateBadges();

document.addEventListener('keydown', function(event) {
    if (currentView !== 'study') return;
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;

    if (event.code === 'Space' && !cardFlipped) {
        event.preventDefault();
        flipCard();
    } else if (cardFlipped && event.key >= '0' && event.key <= '5') {
        event.preventDefault();
        submitGrade(parseInt(event.key));
    }
});
