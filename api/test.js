module.exports = (req, res) => {
  res.status(200).json({ version: 'v3', time: Date.now(), q: req.query });
};
