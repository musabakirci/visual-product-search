"""Add embedding versioning columns to image_embeddings."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_embedding_versioning"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite cannot drop/alter UNIQUE constraints in place, so rebuild the table.
    op.create_table(
        "image_embeddings_new",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column(
            "embedding_version",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("'v1'"),
        ),
        sa.Column(
            "embedding_type",
            sa.String(length=255),
            nullable=False,
            server_default=sa.text("'resnet50'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"]),
        sa.UniqueConstraint(
            "product_id",
            "embedding_version",
            "embedding_type",
            name="uq_image_embeddings_product_id_version_type",
        ),
    )
    # Copy data and explicitly backfill new columns (do not rely on defaults).
    op.execute(
        """
        INSERT INTO image_embeddings_new (
            id,
            product_id,
            model_name,
            embedding_dim,
            embedding,
            cluster_id,
            embedding_version,
            embedding_type,
            created_at
        )
        SELECT
            id,
            product_id,
            model_name,
            embedding_dim,
            embedding,
            cluster_id,
            'v1',
            'resnet50',
            created_at
        FROM image_embeddings
        """
    )
    # Swap tables to preserve the original name.
    op.drop_table("image_embeddings")
    op.rename_table("image_embeddings_new", "image_embeddings")


def downgrade() -> None:
    # Rebuild the original schema (no version/type columns).
    op.create_table(
        "image_embeddings_old",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["product_id"], ["products.id"]),
        sa.UniqueConstraint("product_id"),
    )
    # Collapse multiple rows per product by taking the max(id) per product_id.
    op.execute(
        """
        INSERT INTO image_embeddings_old (
            id,
            product_id,
            model_name,
            embedding_dim,
            embedding,
            cluster_id,
            created_at
        )
        SELECT
            ie.id,
            ie.product_id,
            ie.model_name,
            ie.embedding_dim,
            ie.embedding,
            ie.cluster_id,
            ie.created_at
        FROM image_embeddings AS ie
        INNER JOIN (
            SELECT product_id, MAX(id) AS max_id
            FROM image_embeddings
            GROUP BY product_id
        ) AS latest
        ON ie.product_id = latest.product_id AND ie.id = latest.max_id
        """
    )
    # Swap tables to restore the original name and constraint.
    op.drop_table("image_embeddings")
    op.rename_table("image_embeddings_old", "image_embeddings")
