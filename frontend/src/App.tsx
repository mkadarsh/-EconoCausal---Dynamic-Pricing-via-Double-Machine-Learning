import { useState } from "react";

const discounts = [0, 10, 20] as const;

function App() {
  const [selectedDiscount, setSelectedDiscount] = useState<(typeof discounts)[number]>(0);

  return (
    <main className="app-shell">
      <header>
        <p className="eyebrow">Causal AI for promotional pricing</p>
        <h1>EconoCausal</h1>
        <p className="intro">
          Understand which discount is likely to cause a purchase, rather than
          simply predict who might buy.
        </p>
      </header>

      <section className="card">
        <h2>Discount actions</h2>
        <p className="muted">
          Causal analysis and budget targeting will appear here in the next MVP
          iteration.
        </p>
        <div className="discount-grid" role="group" aria-label="Discount actions">
          {discounts.map((discount) => (
            <button
              className={selectedDiscount === discount ? "discount active" : "discount"}
              key={discount}
              onClick={() => setSelectedDiscount(discount)}
              type="button"
            >
              ${discount}
              <span>discount</span>
            </button>
          ))}
        </div>
        <p className="selection">
          Selected action: <strong>${selectedDiscount} discount</strong>
        </p>
      </section>
    </main>
  );
}

export default App;
