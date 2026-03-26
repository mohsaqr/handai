import { PrismaClient } from '@prisma/client';

// Default to local SQLite if DATABASE_URL is not set (dev, Docker, etc.)
if (!process.env.DATABASE_URL) {
    process.env.DATABASE_URL = "file:./dev.db";
}

const prismaClientSingleton = () => {
    return new PrismaClient();
};

declare global {
    var prisma: undefined | ReturnType<typeof prismaClientSingleton>;
}

const prisma = globalThis.prisma ?? prismaClientSingleton();

export default prisma;

if (process.env.NODE_ENV !== 'production') globalThis.prisma = prisma;
