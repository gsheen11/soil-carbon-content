import matplotlib.pyplot as plt


def scatter(y, y_hat):
    plt.scatter(y, y_hat)

    # Optionally, add a reference line to indicate perfect predictions
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # 'k--' is for black dashed line, 'lw' is line width

    # Label the axes
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Actual vs. Predicted Values')

    # Show the plot
    plt.show()

def residuals(y, y_hat):
    residuals = y - y_hat

    # Create the residual plot
    plt.scatter(y, residuals)

    # Optionally, add a horizontal line at 0 to indicate no residual
    plt.axhline(y=0, color='k', linestyle='--', lw=2)

    # Label the axes
    plt.xlabel('Actual Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    # Show the plot
    plt.show()