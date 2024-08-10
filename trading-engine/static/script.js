function updateData() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            document.getElementById('cash').textContent = data.cash.toFixed(2);
            document.getElementById('position').textContent = data.position;
            const equityElement = document.getElementById('equity');
            if (Array.isArray(data.equity)) {
                equityElement.textContent = data.equity.map(e => e.toFixed(2)).join(', ');
            } else {
                equityElement.textContent = 'N/A';
            }
;

            updateTable('orders', data.orders);
            updateTable('trades', data.trades);
            updateTradeStats(data.trade_stats);
        });
}

function updateTable(tableId, data) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = '';
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'N/A';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function updateTradeStats(stats) {
    const container = document.getElementById('trade-stats');
    container.innerHTML = '';
    Object.entries(stats).forEach(([key, value]) => {
        const p = document.createElement('p');
        p.textContent = `${key.replace(/_/g, ' ').toUpperCase()}: ${typeof value === 'number' ? value.toFixed(2) : value}`;
        container.appendChild(p);
    });
}

// 초기 데이터 로드
updateData();

// 5초마다 데이터 업데이트
setInterval(updateData, 5000);