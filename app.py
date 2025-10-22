from pathlib import Path
import sqlite3
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend

# Configuration
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'static/generated'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'dev-secret-change-me')

DATABASE = os.path.join('instance', 'users.db')
Path('instance').mkdir(exist_ok=True)


def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Auth helpers ---
def create_user(username, password):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                  (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == password:
        return True
    return False


# --- Visualization helpers ---
def save_confusion_matrix_svg(y_true, y_pred, classes, outpath):
    cmatrix = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cmatrix, interpolation='nearest', cmap=cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel='True label', xlabel='Predicted label', title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    thresh = cmatrix.max() / 2.
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(j, i, format(cmatrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cmatrix[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(outpath, format='svg')
    plt.close(fig)


def save_pie_chart_svg(y_pred, outpath):
    vals, counts = np.unique(y_pred, return_counts=True)
    labels = [str(v) for v in vals]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title('Prediction Distribution')
    fig.tight_layout()
    fig.savefig(outpath, format='svg')
    plt.close(fig)


def save_roc_svg(y_true, y_score, classes, outpath):
    y_true_arr = np.array(y_true)
    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    if n_classes == 2:
        if y_score.ndim == 2 and y_score.shape[1] == 1:
            probs = y_score.ravel()
        elif y_score.ndim == 2 and y_score.shape[1] == 2:
            probs = y_score[:, 1]
        else:
            probs = y_score.ravel()
        fpr, tpr, _ = roc_curve(y_true_arr, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    else:
        y_bin = label_binarize(y_true_arr, classes=classes)
        if y_score.shape[1] != len(classes):
            raise ValueError(
                'y_score must have a column per class for multiclass ROC.')
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2,
                label=f'Micro-average ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(outpath, format='svg')
    plt.close(fig)


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html', user=session.get('username'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if create_user(username, password):
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists.', 'danger')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if verify_user(username, password):
            session['username'] = username
            flash('Logged in successfully.', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        flash('Please login to upload CSV.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Only CSV allowed.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        uid = uuid.uuid4().hex
        saved_name = f"{uid}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash('Failed to read CSV: ' + str(e), 'danger')
            return redirect(request.url)

        # --- Detect true labels ---
        y_true = None
        for col in ['y_true', 'true', 'label', 'ground_truth', 'gt', 'actual', 'target', 'class', 'truth']:
            if col in df.columns:
                y_true = df[col].values
                break
        if y_true is None:
            flash(
                'CSV missing a true-label column. Using dummy labels (zeros).', 'warning')
            y_true = np.zeros(len(df), dtype=int)

        # --- Detect predictions ---
        y_pred, y_score = None, None
        if 'y_pred' in df.columns:
            y_pred = df['y_pred'].values

        # --- Detect probability columns ---
        prob_cols = [c for c in df.columns if c.lower().startswith(
            ('prob_', 'y_score', 'score_', 'p_'))]
        if len(prob_cols) >= 1:
            probs = df[prob_cols].values
            if probs.shape[1] == 1:
                y_score = probs
                y_pred = (probs.ravel() >= 0.5).astype(int)
            else:
                y_score = probs
                y_pred = np.argmax(probs, axis=1)

        # --- Handle missing predictions gracefully ---
        if y_pred is None:
            flash('CSV missing predictions. Using dummy predictions (zeros).', 'warning')
            y_pred = np.zeros(len(df), dtype=int)

        # --- Convert non-numeric labels if needed ---
        try:
            y_true = pd.factorize(y_true)[0]
            y_pred = pd.factorize(y_pred)[0]
        except Exception:
            pass

        classes = np.unique(np.concatenate([y_true, y_pred]))
        base_id = uuid.uuid4().hex
        cm_path = os.path.join(GENERATED_FOLDER, f'cm_{base_id}.svg')
        pie_path = os.path.join(GENERATED_FOLDER, f'pie_{base_id}.svg')
        roc_path = os.path.join(GENERATED_FOLDER, f'roc_{base_id}.svg')

        try:
            save_confusion_matrix_svg(y_true, y_pred, classes, cm_path)
            save_pie_chart_svg(y_pred, pie_path)
            if y_score is not None:
                save_roc_svg(y_true, np.asarray(y_score), classes, roc_path)
            else:
                y_score_onehot = label_binarize(y_pred, classes=classes)
                save_roc_svg(y_true, y_score_onehot, classes, roc_path)
        except Exception as e:
            flash('Error generating visualizations: ' + str(e), 'danger')
            return redirect(request.url)

        # --- Compute metrics summary ---
        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(
                y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(
                y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            auc_val = None
            if y_score is not None:
                if len(classes) == 2:
                    probs = y_score[:, 1] if y_score.ndim == 2 and y_score.shape[1] > 1 else y_score.ravel(
                    )
                    auc_val = roc_auc_score(y_true, probs)
                elif y_score.ndim == 2:
                    auc_val = roc_auc_score(label_binarize(y_true, classes=classes),
                                            y_score, average='macro', multi_class='ovr')
            summary = {
                "Samples": len(y_true),
                "Classes": len(classes),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "AUC": round(auc_val, 4) if auc_val is not None else "N/A"
            }
        except Exception as e:
            summary = {"Error": f"Could not compute metrics: {e}"}

        return render_template(
            'results.html',
            user=session.get('username'),
            cm_svg=os.path.relpath(cm_path, start='static'),
            pie_svg=os.path.relpath(pie_path, start='static'),
            roc_svg=os.path.relpath(roc_path, start='static'),
            classes=list(map(str, classes)),
            summary=summary
        )

    return render_template('upload.html', user=session.get('username'))


@app.route('/static/generated/<path:filename>')
def generated(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
